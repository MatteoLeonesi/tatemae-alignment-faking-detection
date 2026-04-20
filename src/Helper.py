import ModelApiClient
def judge_llm(judge_client: ModelApiClient, system_prompt: str, user_prompt: str) -> str:
    final_user_prompt = user_prompt + "\nOnly output exactly one word: Yes or No."

    response = judge_client.call(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_user_prompt}
        ],
    )

    raw = str(response)
    norm = raw.strip().strip('.,:;!?"\'').lower()

    if norm == "yes":
        return "Yes"
    if norm == "no":
        return "No"
    return "No"


def compare_params(toolChoosen: dict, gold: dict) -> bool:
    if not gold:
        return True

    for key, expected_val in gold.items():
        if key not in toolChoosen:
            return False

        pred_val = toolChoosen[key]

        if isinstance(expected_val, bool):
            if isinstance(pred_val, str):
                norm_pred_val = pred_val.lower() == 'true'
            elif isinstance(pred_val, bool):
                norm_pred_val = pred_val
            else:
                return False

            if norm_pred_val != expected_val:
                return False

        elif isinstance(expected_val, str):
            if str(pred_val).lower() != expected_val.lower():
                return False

        else:
            if pred_val != expected_val:
                return False

    return True
