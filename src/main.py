import argparse
from pathlib import Path

from PipelineRunner import PipelineRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tatemae potential-AF study")
    parser.add_argument("--controls", action="store_true", help="Add Call 2′ and evaluation_only (two extra model calls per scenario).")
    parser.add_argument("--data", default=str(Path(__file__).with_name("dataset.json")))
    parser.add_argument("--output", help="Destination CSV; controls use a separate default filename.")
    args = parser.parse_args()
    runner = PipelineRunner(
        exp_provider="openrouter",  # model provider (openrouter: MAX_WORKERS=5; local: MAX_WORKERS=1) PipelineRunner.py
        exp_model="deepseek/deepseek-v3.2",   #
        judge_provider="openai",
        judge_model="gpt-5.4-mini",
        data_path=args.data,
        output_path=args.output or ("alignment_faking_controls.csv" if args.controls else "alignment_faking_evaluated.csv"),
        controls=args.controls
    )
    runner.run()
