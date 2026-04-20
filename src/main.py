from PipelineRunner import PipelineRunner

if __name__ == "__main__":
    runner = PipelineRunner(
        exp_provider="openrouter",  # model provider (openrouter: MAX_WORKERS=5; local: MAX_WORKERS=1) PipelineRunner.py
        exp_model="deepseek/deepseek-v3.2",   #
        judge_provider="openai",
        judge_model="gpt-5.4-mini"
    )
    runner.run()
