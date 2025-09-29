import re
import ast
import string
from openai import OpenAI
from collections import Counter
import numpy as np


openai_api_key = "EMPTY"
openai_api_base = "http://your_service_ip:port/v1"

client = OpenAI(
    api_key = openai_api_key,
    base_url = openai_api_base,
)

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def text_has_answer(answers, text) -> bool:
    if isinstance(answers, str):
        answers = [answers]
    text = normalize_answer(text)
    for single_answer in answers:
        single_answer = normalize_answer(single_answer)
        if single_answer in text:
            return True
    return False


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def get_ans(prompt):
    
    chat_response = client.chat.completions.create(
        model="qwen2.5_14b_instruct",  
        messages=[
            {"role": "system", "content": "You are a precise Q&A assistant."},
            {"role": "user", "content": prompt}
        ],
        stream=False,
        temperature=0,
        top_p=0.001,
        max_tokens=128,
    )
    ans = chat_response.choices[0].message.content
    # print(ans)
    return ans


def build_prompt_nq(summary, question):
    prompt = f"""
[Instruction] Answer the question. 
IMPORTANT: Respond ONLY with the exact answer in the same format as the examples. 
Do NOT add any extra text, explanations, or punctuation. Do NOT include "Answer:" or any similar prefix in your response.

[Examples]
Question: who is the woman washing the car in cool hand luke
Answer: Joy Harmon

Question: who said it's better to have loved and lost
Answer: Alfred , Lord Tennyson

Question: name the first indian woman to be crowned as miss world
Answer: Reita Faria 

[Current Question]
{summary} 
Question: {question}
Answer:
"""
    return prompt.strip()


def build_prompt_hotpot(summary, question):
    prompt = f"""
[Instruction] Answer the question. 
IMPORTANT: Respond ONLY with the exact answer in the same format as the examples. 
Do NOT add any extra text, explanations, or punctuation. Do NOT include "Answer:" or any similar prefix in your response.

[Examples]
Question: Which magazine was started first Arthur's Magazine or First for Women?
Answer: Arthur's Magazine

Question: The Oberoi family is part of a hotel company that has a head office in what city?
Answer: Delhi

Question: Musician and satirist Allie Goertz wrote a song about the ""The Simpsons"" character Milhouse, who Matt Groening named after who?
Answer: President Richard Nixon

[Current Question]
{summary} 
Question: {question}
Answer:
"""
    return prompt.strip()


def extract_answer(answer):
    if "im_end" in answer:
        answer = answer.replace("<|im_end|>", "")
    return answer.strip()

def compute_score_nq(data_source, solution_str, ground_truth, extra_info):
    solution_str = extract_answer(solution_str)
    question = extra_info["question"]
    endtask_input = build_prompt_nq(solution_str, question)
    endtask_llm_output = get_ans(endtask_input)
    
    if isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.tolist()  
    elif not isinstance(ground_truth, list):
        ground_truth = ast.literal_eval(ground_truth)

    em_reward = any([exact_match(endtask_llm_output, answer) for answer in ground_truth])
    
    # Combine EM and F1 scores (you can adjust the weights as needed)
    combined_score = em_reward * 10 
    
    
    return combined_score


def compute_score_hotpot(data_source, solution_str, ground_truth, extra_info):
    solution_str = extract_answer(solution_str)
    question = extra_info["question"]
    endtask_input = build_prompt_hotpot(solution_str, question)
    endtask_llm_output = get_ans(endtask_input)
    
    if isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.tolist()  
    elif not isinstance(ground_truth, list):
        ground_truth = ast.literal_eval(ground_truth)

    em_reward = any([exact_match(endtask_llm_output, answer) for answer in ground_truth])

    
    # Combine EM and F1 scores (you can adjust the weights as needed)
    combined_score = em_reward * 10 
  
    return combined_score