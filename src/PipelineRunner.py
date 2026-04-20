import json

import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from ExperimentingRow import ExperimentRow
from ModelApiClient import ModelApiClient
from Prompts import SYSTEM_PROMPT


class PipelineRunner:
    def __init__(self,
                 exp_provider: str,
                 exp_model: str,
                 judge_provider: str,
                 judge_model: str,
                 data_path: str = "dataset.json",
                 output_path: str = "alignment_faking_evaluated.csv"):

        self.SYSTEM_PROMPT = SYSTEM_PROMPT
        self.DATA_PATH = data_path
        self.OUTPUT_PATH = output_path

        self.exp_client = ModelApiClient(exp_provider, exp_model)
        self.judge_client = ModelApiClient(judge_provider, judge_model)

    def _load_data(self) -> list[dict]:
        try:
            with open(self.DATA_PATH, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"ERROR: {self.DATA_PATH}")
            return []

    def _format_results(self, row: ExperimentRow, original_data: dict) -> dict:
        return {
            'id': original_data['id'],
            'category': row.category,
            'adversarial_instruction_prompt_type': row.adversarial_instruction_prompt_type,
            'gold_truth_tool': row.gold_tool,
            'af_tool_param_full': 'Yes' if row.af_tool_param_full else 'No',

            'judge_R3_AF_C1': row.judge_R3_AF_C1,
            'judge_R3_AF_C2': row.judge_R3_AF_C2,
            'judge_R3_AF_C3': row.judge_R3_AF_C3,

            'T1_full_correct': 'Yes' if row.T1_full_correct else 'No',
            'call1_tool (T_1)': row.T1,
            'call1_params (P_1)': json.dumps(row.P1),
            'Y0_Um (Y_0)': row.Y0_Um,
            'call1_raw_response': row.raw_Call1,

            'T2_full_correct': 'Yes' if row.T2_full_correct else 'No',
            'judge_R2_vs_Um': row.judge_R2_vs_Um,
            'call2_think (R_2)': row.R2,
            'call2_tool (T_2)': row.T2,
            'cal2_params (P_2)': json.dumps(row.P2),
            'call2_output (Y_2)': row.Y2,
            'call2_raw_response': row.raw_Call2,

            'T3_full_correct': 'Yes' if row.T3_full_correct else 'No',
            'judge_R3_vs_Um': row.judge_R3_vs_Um,
            'call3_think (R_3)': row.R3,
            'call3_tool (T_3)': row.T3,
            'cal3_params (P_3)': json.dumps(row.P3),
            'call3_output (Y_3)': row.Y3,
            'call3_raw_response': row.raw_Call3,
        }

    def _process_row(self, original_data: dict) -> dict | None:
        try:
            row = ExperimentRow(original_data)

            row.run_experiment_calls(self.exp_client, self.SYSTEM_PROMPT)
            row.calculate_full_correctness()
            row.run_judge_calls(self.judge_client, self.SYSTEM_PROMPT)
            row.calculate_af_metric()

            return self._format_results(row, original_data)

        except Exception as e:
            row_id = original_data.get('id', 'unknown')
            print(f"\n[ERROR] Failed to process row {row_id}: {e}")
            return None

    def run(self):
        data_rows = self._load_data()
        if not data_rows:
            return

        MAX_WORKERS = 5 # we don't suggest to increase it

        all_results = []
        print(f"Starting parallel processing with max {MAX_WORKERS} workers...")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results_iterator = executor.map(self._process_row, data_rows)

            for result in tqdm(results_iterator,
                               total=len(data_rows),
                               desc="Running Alignment Faking Experiment"):

                if result:
                    all_results.append(result)

        df = pd.DataFrame(all_results)
        df.to_csv(self.OUTPUT_PATH, index=False)
        print(f"\n done {len(all_results)} - {self.OUTPUT_PATH}")

        total_count = len(df)
        af_count = (df['af_tool_param_full'] == 'Yes').sum()
        af_percentage = (af_count / total_count * 100) if total_count > 0 else 0.0
        print(f"--- AF (Tool + Param) Count: {af_count}/{total_count} ({af_percentage:.2f}%) ---")

        print("\n--- Detailed Alignment Faking Breakdown for R3---")
        af_rows = df[df['af_tool_param_full'] == 'Yes']
        not_af_rows = df[df['af_tool_param_full'] == 'No']

        criteria = ['judge_R3_AF_C1', 'judge_R3_AF_C2', 'judge_R3_AF_C3']

        for name, data in [("AF Rows", af_rows), ("Non-AF Rows", not_af_rows)]:
            if data.empty: continue
            print(f"\n--- {name} Score Breakdown ---")

            scores = (data[criteria] == 'Yes').sum(axis=1)
            counts = scores.value_counts().sort_index()

            for i in range(4):
                print(f"Score {i}/3: {counts.get(i, 0)}")