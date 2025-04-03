import re
from typing import Optional
import datasets


def mazenav_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def mazenav_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["text"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def mazenav_process_results(doc, results):
    pred = results[0]

    question_id = parse_question_id(doc["id"])
    pred = extract_answer_from_text_mazenav(pred, question_id)

    ref_ans = str(doc["oracle_answer"]).lower()
    ref_ans_opt = str(doc["oracle_option"]).lower()
    pred = str(pred).lower() if pred is not None else "null"
    eval_result = int(ref_ans.lower() in pred.lower())
    eval_result_opt = int(ref_ans_opt.lower() in pred.lower())
    eval_result = max(eval_result, eval_result_opt)
    # eval_summary.append({"ref": ref_ans, "model_output": pred, "eval_result": eval_result})

    # score = relaxed_correctness(pred, doc["answer"])
    # score = 1.0 if score else 0.0
    return_dict = {"parsed_match": eval_result}
    # if type == "human_test":
    #     return_dict["relaxed_human_split"] = score
    # else:
    #     return_dict["relaxed_augmented_split"] = score
    return return_dict


def mazenav_process_docs(ds: datasets.Dataset) -> datasets.Dataset:
    ds = ds.filter(lambda x: "mazenav" in x["id"])
    return ds


def parse_question_id(id: str) -> int:
    question_id = id.split(".")[-1]
    return int(question_id)


def extract_answer_from_text_mazenav(text: str, question_id: int = 0) -> Optional[str]:
    """Extracts answers from maze navigation text based on the question ID."""
    number_mapping = {"zero": 0, "no": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}

    if question_id == 2:  # require binary answers
        yes_patterns = [r"\byes\b", r"the answer is yes", r"\"yes\"", r"\'yes\'", r"is the shortest path"]
        no_patterns = [r"\bno\b", r"the answer is no", r"\"no\"", r"\'no\'", r"\bnot\b"]

        # Check for "Yes" answers
        for pattern in yes_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return "Yes"

        # Check for "No" answers
        for pattern in no_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return "No"

    else:
        # Check for textual number patterns first
        for text_num, num in number_mapping.items():
            pattern = rf"\b{text_num}\b"
            if re.search(pattern, text, re.IGNORECASE):
                return num

        patterns = {
            0: [  # For right turns
                r"\bThere are\s*(\d+)\s*right turns\b",  # for proprietary
                r"\bThere is\s*(\d+)\s*right turn\b",
                r"\b(\d+)\s+right turn(s)?",
                r"answer is\s+(\d+)",
                r"answer is:\s*\n*\s*(\d+)",
                r"from S to E is\s+(\d+)",
                r"Answer:\*\*\s*(\d+)\b",
            ],
            1: [  # For total turns
                r"\bThere are\s*(\d+)\s*total turns\b",  # for proprietary
                r"\bThere are\s*(\d+)\s*turns\b",
                r"There is\s*(\d+)\s*turn\b",
                r"There is\s*(\d+)\s*total turn\b",
                r"answer is\s+(\d+)",
                r"answer is:\s*\n*\s*(\d+)",
                r"from S to E is\s+(\d+)",
                r"\btotal of\s+(\d+)\s+turn(s)?",
                r"Answer:\*\*\s*(\d+)\b",
                r"(\d+)\s+total turn(s)?",
            ],
        }

        for pattern in patterns[question_id]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))  # Return the first matching group as integer

        # If no specific pattern matches, try to extract the first number in the text
        fallback_match = re.search(r"\d+", text)
        if fallback_match:
            return int(fallback_match.group(0))
        # fallback_match_list = re.findall(r"\d+", text)
        # if fallback_match_list:
        #     return int(fallback_match_list[-1])

    return None  # Return None if no number or textual number is found at all


def relaxed_correctness(prediction, target, max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    This funcion is taken from https://github.com/QwenLM/Qwen-VL/blob/34b4c0ee7b07726371b960911f249fe61b362ca3/eval_mm/evaluate_vqa.py#L113
    Args:
      target: List of target string.
      prediction: List of predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str):
        try:
            if text.endswith("%"):
                # Convert percentages to floats.
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()
