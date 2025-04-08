import os
import re
from typing import Optional

try:
    import openai
except ImportError:
    raise ImportError("openai is not installed. Please install it to use this function.")

try:
    from jinja2 import Environment, FileSystemLoader
except ImportError:
    raise ImportError("jinja2 is not installed. Please install it to use this function.")


def frozenlake_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def load_prompt_from_path(prompt_path: str) -> str:
    pass


def frozenlake_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    question = question + "\nAvailable options: " + str(doc["choices"]) + "."

    if "prompt_path" in lmms_eval_specific_kwargs:
        env = Environment(loader=FileSystemLoader(""))
        template = env.get_template(lmms_eval_specific_kwargs["prompt_path"])
        return template.render({"question": question})

    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def judge_with_llm(pred, ref, lmmms_eval_specific_kwargs):
    # This function is not used in the frozenlake task
    # but is required for the interface to be consistent with other tasks.
    api_key = os.getenv("JUDGE_LLM_API_KEY")
    if api_key is None:
        raise ValueError("LLM_JUDGE_API_KEY is not provided in the environment variables.")

    if "judge_llm_base_url" in lmmms_eval_specific_kwargs:
        base_url = lmmms_eval_specific_kwargs["base_url"]
    else:
        base_url = os.getenv("JUDGE_LLM_API_BASE_URL")
        if base_url is None:
            raise ValueError("base_url is not provided in the arguments or environment variables.")

    if "judge_llm_model" in lmmms_eval_specific_kwargs:
        judge_llm = lmmms_eval_specific_kwargs["judge_llm"]
    else:
        judge_llm = os.getenv("JUDGE_LLM_MODEL")
        if judge_llm is None:
            raise ValueError("judge_llm is not provided in the arguments or environment variables.")

    if "judge_llm_generation_kwargs" in lmmms_eval_specific_kwargs:
        judge_llm_generation_kwargs = lmmms_eval_specific_kwargs["judge_llm_generation_kwargs"]
    else:
        judge_llm_generation_kwargs = dict()

    client = openai.Client(api_key=api_key, base_url=base_url)

    judge_prompt = """
You will be given two inputs:
1.	Model Prediction: This contains both the reasoning steps and the final predicted answer of a model.
2.	Ground Truth Answer: This is the correct final answer, given as a single integer.

Your task is to determine whether the model's final answer is correct by comparing it with the ground truth answer.

Inputs:

Model Prediction:
{model_prediction}

Ground Truth Answer:
{ground_truth_answer}

Instructions:
* Read the model's reasoning and identify the final answer it gives.
* Compare this final answer with the ground truth.
* Respond with one of the following outputs only:
Correct
Incorrect

Do not explain your decision. Only return Correct or Incorrect. 
"""

    response = client.chat.completions.create(
        model=judge_llm,
        messages=[{"role": "user", "content": judge_prompt.format(model_prediction=pred, ground_truth_answer=ref)}],
        temperature=0,
        max_tokens=512,
    )

    # Extract the model's response
    judge_response = response.choices[0].message.content.strip()
    print(f"Judge response: {judge_response}")
    if "incorrect" in judge_response.lower():
        print("Judge response: Incorrect")
        return 0
    print("Judge response: Correct")
    return 1


def frozenlake_process_results(doc, results):
    pred = results[0]

    # question_id = parse_question_id(doc["id"])
    # llm_judge_result = judge_with_llm(pred, doc["answer"], dict())

    parsed_pred = extract_answer_from_text_frozenlake(pred, question_id=0)  # HARD CODED QUESTION ID

    ref_ans = str(doc["answer"]).lower()
    parsed_pred = str(parsed_pred).lower() if parsed_pred is not None else "null"
    eval_result = int(ref_ans.lower() in parsed_pred.lower())
    # eval_result_opt = int(ref_ans_opt.lower() in pred.lower())
    # eval_result = max(eval_result, eval_result_opt)

    # return_dict = {"parsed_match": eval_result, "llm_match": llm_judge_result}
    return_dict = {"parsed_match": eval_result}
    return return_dict


def parse_question_id(id: str) -> int:
    question_id = id.split(".")[-1]
    return int(question_id)


def extract_answer_from_text_frozenlake(text: str, question_id: int = 0) -> Optional[str]:
    """Extracts answers from frozenlake navigation text based on the question ID."""
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
            3: [  # For left turns
                r"\bThere are\s*(\d+)\s*left turns\b",  # for proprietary
                r"\bThere is\s*(\d+)\s*left turn\b",
                r"\b(\d+)\s+left turn(s)?",
                r"answer is\s+(\d+)",
                r"answer is:\s*\n*\s*(\d+)",
                r"from S to E is\s+(\d+)",
                r"Answer:\*\*\s*(\d+)\b",
            ],
        }

        for p in patterns.values():
            for pattern in p:
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
