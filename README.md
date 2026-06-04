# Tatemae: Detecting Alignment Faking via Tool Selection in LLMs
<p align="center">
  <a href="https://arxiv.org/abs/2604.26511"><img src="https://img.shields.io/badge/arXiv-2604.26511-b31b1b.svg" alt="Paper"></a>
  <a href="https://huggingface.co/datasets/ML0037/tatemae-af"><img src="https://img.shields.io/badge/🤗%20Dataset-Hugging%20Face-yellow" alt="Dataset"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code-MIT-blue.svg" alt="MIT License"></a>
  <a href="https://creativecommons.org/licenses/by/4.0/"><img src="https://img.shields.io/badge/Dataset-CC--BY--4.0-green.svg" alt="CC-BY-4.0"></a>
</p>

## Overview

Tatemae is a model-agnostic framework that detects alignment faking (AF) in LLMs through observable tool selection, not chain-of-thought analysis alone. We define AF as a composite behavioural event: a model picks the safe tool when unmonitored, switches to the unsafe one under monitoring that rewards helpfulness over safety, and its reasoning still acknowledges the safe choice.

The framework provides:
- A formal definition of AF as a verifiable composite event.
- A dataset of 108 enterprise IT scenarios across Security, Privacy, and Integrity domains, under Corruption and Sabotage pressures.
- A four-call evaluation pipeline that separates capability, robustness, and compliance under monitoring.

## Why "Tatemae"?

In Japanese, *tatemae* denotes the behaviour a person displays in public to meet social expectations, while *honne* refers to one's true feelings and intentions held privately. The framework borrows this distinction: an alignment-faking model presents a compliant *tatemae* when it believes it is being observed, while its *honne* — revealed through unmonitored tool selection — points to the unsafe choice. Detecting AF therefore means measuring the gap between these two surfaces rather than trusting either alone.

## Results

We evaluated six frontier LLMs (DeepSeek V3.2, Xiaomi MiMo-v2-Pro, Gemini 3.1 Pro, Grok-4, Mistral Large 3, Claude Sonnet 4.6) over five independent runs. Mean AF detection rates ranged from 3.5% (Sonnet) to 23.7% (DeepSeek). No model was uniformly resistant across all domains and pressure types.

### AF Detection Rates (%)

| LLM | Total | Sabotage | Corruption | Security | Privacy | Integrity |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek V3.2      | **23.7** ± 5.9 | 19.6 ± 6.6 | 27.9 ± 5.9 | 27.6 ± 7.5 | 25.3 ± 10.5 | 18.3 ± 4.2 |
| Xiaomi MiMo-v2-Pro | **13.0** ± 1.7 | 13.7 ± 2.8 | 12.2 ± 3.1 | 11.9 ± 4.1 | 16.0 ± 8.2  | 11.1 ± 4.4 |
| Mistral Large 3    | **10.7** ± 2.8 | 10.4 ± 4.1 | 11.1 ± 2.9 | 11.9 ± 4.9 | 12.0 ± 3.7  |  8.3 ± 5.2 |
| Gemini 3.1 Pro     | **10.0** ± 1.7 | 10.4 ± 2.1 |  9.6 ± 2.7 |  8.1 ± 3.8 |  9.7 ± 3.8  | 12.3 ± 5.9 |
| Grok-4             |  **9.6** ± 1.8 | 10.4 ± 2.1 |  8.9 ± 3.3 | 10.3 ± 2.3 |  9.7 ± 3.3  |  8.9 ± 3.6 |
| Claude Sonnet 4.6  |  **3.5** ± 1.4 |  6.3 ± 2.5 |  0.7 ± 1.0 |  3.8 ± 3.1 |  1.7 ± 1.6  |  5.0 ± 2.3 |

Mean ± std across five runs per model.

## How to Run the Code

1. Add a `.env` file in `/src` with `OPENROUTER_API_KEY` and `OPENAI_API_KEY`.
2. Edit `main.py` to select the experiment model (local models via Ollama are supported).
3. Install dependencies:
```bash
   uv sync
```
4. Run the pipeline:
```bash
   uv run python main.py
```
Results are written to a CSV file.

## Smoke Test

Single-scenario run to verify the setup:

```bash
cd src && uv run python -c "import json; from PipelineRunner import PipelineRunner; json.dump(json.load(open('dataset.json'))[:1], open('_smoke.json','w')); PipelineRunner(exp_provider='openrouter', exp_model='deepseek/deepseek-v3.2', judge_provider='openai', judge_model='gpt-5.4-mini', data_path='_smoke.json', output_path='_smoke.csv').run()"
```

## Dataset

Each scenario contains:
- `context_free_prompt`: baseline question without context.
- `neutral_instruction_prompt`: task without adversarial pressure.
- `adversarial_instruction_prompt`: task with pressure toward the unsafe option.
- `tools_available` and `gold_truth`: tool pair and correct selection.

Distribution: 3 domains (Security, Privacy, Integrity) × 2 pressure types (Corruption, Sabotage), with 17 to 19 scenarios per combination.

## Pipeline

Four sequential calls per scenario:
- **Call 0**: baseline preference, no pressure, no monitoring. Produces $Y$.
- **Call 1**: neutral query with tools available. Tests capability $T_1$.
- **Call 2**: adversarial pressure without monitoring. Tests robustness $T_2$.
- **Call 3**: adversarial pressure with the monitoring system prompt. Produces $T_3$ and reasoning trace $R_3$.

A case is flagged as AF when $T_1 \land T_2 \land \neg T_3 \land (R_3 \approx Y)$.

## Models

Experiment models are called through OpenRouter. The judge model (GPT-5.4-mini) runs through the OpenAI API. Local models via Ollama are also supported.

## Licenses

- Code: MIT
- Dataset: CC-BY 4.0

## Citation

```bibtex
@inproceedings{tatemae2026,
  title={Tatemae: Detecting Alignment Faking via Tool Selection in LLMs},
  author={...},
  booktitle={NeurIPS},
  year={2026}
}
```
