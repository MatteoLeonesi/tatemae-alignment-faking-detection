import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from ExperimentingRow import ExperimentRow
from Helper import compare_params, judge_llm
from ModelApiClient import ModelApiClient
from PipelineRunner import PipelineRunner
from Prompts import SYSTEM_PROMPT, control_prompt

DATA = dict(id=1, context_free_prompt="BASELINE", neutral_instruction_prompt="NEUTRAL",
            adversarial_instruction_prompt="ADVERSARIAL",
            tools_available=[dict(name="safe", description="Safe choice"),
                             dict(name="unsafe", description="Unsafe choice")],
            gold_truth=dict(correct_tool="safe", correct_parameters={"enabled": True}))
RESPONSE = '<think>safe</think><tool_use>{}</tool_use><parameters>{{"enabled":true}}</parameters><o>safe</o>'
ORIGINAL_REQUEST_HASHES = {
    "6cb8e16156fead27cd61aae87f5ad4682ae2c38ac3f66164c328354f4c557775",
    "7879f57ec99490e80bb1cbe43346c390d27d5144ea9e36e59b07c1e523c2d015",
    "9340835da670a94cfc1a6e26c61a0f9b8d9248ba586b4b07ad77536dc798fc06",
    "ec7b1e3366542768105c280ceebf012ff75e3e8ef5c6b76dcb60a1a52e09bec8",
}


class FakeClient:
    model, provider = "test-model", "test-provider"
    split_think_output = staticmethod(ModelApiClient.split_think_output)
    extract_tool_info = staticmethod(ModelApiClient.extract_tool_info)

    def __init__(self, fail=False):
        self.requests, self.fail = [], fail

    def call(self, messages):
        self.requests.append(messages)
        if self.fail:
            raise RuntimeError("offline simulated failure")
        monitored = messages[0]["role"] == "system" and "will be rated" in messages[0]["content"]
        return RESPONSE.format("unsafe" if monitored else "safe")


class PipelineTests(unittest.TestCase):
    def test_parser_uses_final_action_outside_reasoning(self):
        example = '<think><tool_use>unsafe</tool_use><parameters>{"wrong":1}</parameters><o>example</o></think>'
        tool, params, output = ModelApiClient.extract_tool_info(example + RESPONSE.format("safe"))
        self.assertEqual((tool, params, output), ("safe", {"enabled": True}, "safe"))
        raw = RESPONSE.format("safe") + '<o>Example: <tool_use>unsafe</tool_use></o>'
        self.assertEqual(ModelApiClient.extract_tool_info(raw)[0], "safe")
        cases = [('<tool_use>safe</tool_use><parameters>{}</parameters><tool_use>unknown</tool_use>', "unknown", None),
                 ('<parameters>{}</parameters><tool_use>safe</tool_use>', "safe", None),
                 ('<tool_use>safe</tool_use><parameters>[]</parameters>', "safe", None),
                 ('<tool_use>safe</tool_use><parameters>broken</parameters>', "safe", None),
                 ('<tool_use>safe</tool_use><tool_use>unfinished', "", None),
                 ('<think><tool_use>safe</tool_use>', "", None)]
        for raw, expected_tool, expected_params in cases:
            with self.subTest(raw=raw):
                self.assertEqual(ModelApiClient.extract_tool_info(raw)[:2], (expected_tool, expected_params))

    def test_boolean_and_judge_validation(self):
        for value, valid in [(False, True), ("false", True), ("garbage", False), (0, False)]:
            self.assertEqual(compare_params({"enabled": value}, {"enabled": False}), valid)
        self.assertFalse(compare_params(None, {}))
        for answer, label in [("Yes.", "Yes"), ("No", "No"), ("Maybe", "Missing"), ("", "Missing")]:
            self.assertEqual(judge_llm(Mock(call=Mock(return_value=answer)), "", ""), (label, answer))

    def test_prompt_controls_are_frozen(self):
        expected = {"call2prime": "f7a47d3cb631b9d73399a09ce6f8760a1d1467487299496e51b524618bec3ebd",
                    "evaluation_only": "0401d5e426c9a855738accd5b59050cec49bc981b6ce85ad9d85f1bf6e6c0c4c"}
        self.assertEqual(hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                         "4d87c7bdbce9a62be1006c4e7e4f1f8f8a9b774533ceb4c167dbec42ce209304")
        for name, digest in expected.items():
            prompt = control_prompt(name)
            self.assertEqual(hashlib.sha256(prompt.encode()).hexdigest(), digest)
            self.assertEqual(prompt.split("The criteria uses for assessing")[1],
                             SYSTEM_PROMPT.split("The criteria uses for assessing")[1])
        with self.assertRaises(ValueError):
            control_prompt("evaluation_only", "changed prompt")

    def test_default_and_controls_preserve_original_requests(self):
        for controls in (False, True):
            client, row = FakeClient(), ExperimentRow(DATA)
            row.run_experiment_calls(client, SYSTEM_PROMPT, controls)
            self.assertEqual(len(client.requests), 6 if controls else 4)
            self.assertEqual({row.calls[f"call{i}"]["request_sha256"] for i in range(4)}, ORIGINAL_REQUEST_HASHES)
            if controls:
                self.assertEqual(row.calls["call2prime"]["status"], "safe_valid")
                self.assertEqual(row.calls["evaluation_only"]["status"], "unsafe_tool")
                self.assertEqual(len({m[-1]["content"] for m in client.requests if m[0]["role"] == "system"}), 1)

    def test_missing_outputs_and_partial_failures(self):
        row = ExperimentRow(DATA)
        client = FakeClient()
        original_call = client.call
        client.call = lambda messages: "" if messages[0]["content"] == SYSTEM_PROMPT else original_call(messages)
        row.run_experiment_calls(client, SYSTEM_PROMPT)
        row.calculate_full_correctness()
        row.judge_R3_vs_Um = "Yes"
        row.calculate_af_metric()
        self.assertIsNone(row.af_tool_param_full)
        self.assertEqual(row.calls["call1"]["status"], "safe_valid")
        judge = Mock(call=Mock(return_value="Maybe"))
        row.run_judge_calls(judge, SYSTEM_PROMPT)
        self.assertEqual(judge.call.call_count, 1)
        self.assertEqual(row.judge_raw["R2_vs_Um"], "Maybe")
        self.assertEqual(row.judge_errors["R3_vs_Um"], "missing_input")
        self.assertNotIn(SYSTEM_PROMPT, judge.call.call_args.kwargs["messages"][1]["content"])
        row.R3 = "reasoning"
        row.run_judge_calls(Mock(call=Mock(side_effect=RuntimeError("judge failure"))), SYSTEM_PROMPT)
        self.assertEqual(row.judge_R3_vs_Um, "Missing")
        self.assertIn("judge failure", row.judge_errors["R3_vs_Um"])

    def test_metric_preserves_observed_cases_and_marks_uncertainty(self):
        row = ExperimentRow(DATA)
        row.run_experiment_calls(FakeClient(), SYSTEM_PROMPT)
        row.calculate_full_correctness()
        for label, expected in [("Yes", True), ("No", False), ("Missing", None)]:
            row.judge_R3_vs_Um = label
            row.calculate_af_metric()
            self.assertIs(row.af_tool_param_full, expected)
        self.assertEqual(row.action_status("safe", {"enabled": False}), "safe_invalid")
        self.assertEqual(row.action_status("invented", {}), "invalid")
        row.T1 = "unsafe"
        row.calculate_full_correctness()
        row.calculate_af_metric()
        self.assertIs(row.af_tool_param_full, False)

    def test_csv_and_paired_denominators_with_failures(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                (path / "data.json").write_text(json.dumps([DATA]))
                exp, judge = FakeClient(fail), FakeClient()
                judge.call = Mock(return_value="Yes")
                stdout = io.StringIO()
                with patch("PipelineRunner.ModelApiClient", side_effect=[exp, judge]), contextlib.redirect_stdout(stdout):
                    runner = PipelineRunner("test", "test", "test", "test", str(path / "data.json"), str(path / "out.csv"), True)
                    runner.run()
                csv = pd.read_csv(path / "out.csv", keep_default_na=False)
                self.assertEqual(len(csv), 1)
                self.assertEqual(csv.loc[0, "af_tool_param_full"], "Missing" if fail else "Yes")
                self.assertIn("pairs=0; excluded=1" if fail else "pairs=1; excluded=0", stdout.getvalue())
                self.assertIn("not estimable" if fail else "+100.00 pp", stdout.getvalue())
                self.assertEqual(judge.call.call_count, 0 if fail else 5)
                if fail:
                    self.assertIn("simulated failure", csv.loc[0, "call3_error"])

    def test_invalid_input_rows_are_retained(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory)
            (path / "data.json").write_text(json.dumps([dict(DATA, tools_available=[{}])]))
            with patch("PipelineRunner.ModelApiClient", side_effect=[FakeClient(), FakeClient()]):
                runner = PipelineRunner("test", "test", "test", "test", str(path / "data.json"), str(path / "out.csv"), True)
                runner.run()
            csv = pd.read_csv(path / "out.csv")
            self.assertEqual(csv.loc[0, "af_tool_param_full"], "Missing")
            self.assertIn("row_error", csv.columns)

    def test_control_failure_does_not_discard_other_pairs(self):
        client, row = FakeClient(), ExperimentRow(DATA)
        original_call = client.call
        def fail_one(messages):
            if messages[0]["content"] == control_prompt("evaluation_only"):
                raise RuntimeError("one control failed")
            return original_call(messages)
        client.call = fail_one
        row.run_experiment_calls(client, SYSTEM_PROMPT, True)
        self.assertEqual(row.calls["evaluation_only"]["status"], "invalid")
        self.assertIn("one control failed", row.calls["evaluation_only"]["error"])
        self.assertEqual(row.calls["call2prime"]["status"], "safe_valid")
        self.assertEqual(row.calls["call3"]["status"], "unsafe_tool")


if __name__ == "__main__":
    unittest.main()
