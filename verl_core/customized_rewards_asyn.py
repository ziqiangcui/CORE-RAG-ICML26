import re
from verl.utils.reward_score.api_generative import prime_query_openai_async # api
import torch
import ast
import numpy as np
import string
import re
import ast
import string
from openai import OpenAI
from collections import Counter
import numpy as np


def compute_score_nq(config, solution_str, ground_truth, data_source, extra_info): 
    solution_str = extract_answer(solution_str)
    # print('rollout result: ', sequences_str)
    question = extra_info["question"]
    endtask_input = build_prompt_nq(solution_str, question)
    # endtask_llm_output = get_ans(endtask_input)
    
    if isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.tolist()  
    elif not isinstance(ground_truth, list):
        ground_truth = ast.literal_eval(ground_truth)
    
    # reward model
    resp = call_api_reward_model(config, endtask_input, ground_truth,data_source, extra_info)
    # print('debug resp:', resp)
    endtask_llm_output = resp.json()['choices'][0]['message']['content']
    # print('reward model response:', endtask_llm_output)

    em_reward = any([exact_match(endtask_llm_output, answer) for answer in ground_truth])
    
    # Combine EM and F1 scores (you can adjust the weights as needed)
    f1_reward = max([f1_score(endtask_llm_output, answer) for answer in ground_truth])
    
    # has_answer_reward = int(text_has_answer(ground_truth, solution_str))

    combined_score = em_reward * 10 + f1_reward * 5   # Example weights: EM is more important
    
    
    return combined_score


def compute_score_tqa(config, solution_str, ground_truth, data_source, extra_info): 
    solution_str = extract_answer(solution_str)
    # print('rollout result: ', sequences_str)
    question = extra_info["question"]
    endtask_input = build_prompt_tqa(solution_str, question)
    # endtask_llm_output = get_ans(endtask_input)
    # print("ground_truth value:", ground_truth) 
    # print("ground_truth type:", type(ground_truth))  
    if isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.tolist()  
    elif not isinstance(ground_truth, list):
        ground_truth = ast.literal_eval(ground_truth)
    
    # reward 
    resp = call_api_reward_model(config, endtask_input, ground_truth,data_source, extra_info)
    # print('debug resp:', resp)
    endtask_llm_output = resp.json()['choices'][0]['message']['content']
    # print('reward model response:', endtask_llm_output)

    em_reward = any([exact_match(endtask_llm_output, answer) for answer in ground_truth])
    
    f1_reward = max([f1_score(endtask_llm_output, answer) for answer in ground_truth])
    
    # has_answer_reward = int(text_has_answer(ground_truth, solution_str))

    combined_score = em_reward * 10 + f1_reward * 10   # Example weights: EM is more important
    
    
    return combined_score


def compute_score_hotpotqa(config, solution_str, ground_truth, data_source, extra_info): 
    solution_str = extract_answer(solution_str)
    # print('rollout result: ', sequences_str)
    question = extra_info["question"]
    endtask_input = build_prompt_hotpotqa(solution_str, question)
   
    if isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.tolist()  
    elif not isinstance(ground_truth, list):
        ground_truth = ast.literal_eval(ground_truth)
    
    # reward model
    resp = call_api_reward_model(config, endtask_input, ground_truth,data_source, extra_info)
    # print('debug resp:', resp)
    endtask_llm_output = resp.json()['choices'][0]['message']['content']

    em_reward = any([exact_match(endtask_llm_output, answer) for answer in ground_truth])
    
    f1_reward = max([f1_score(endtask_llm_output, answer) for answer in ground_truth])
    
    # has_answer_reward = int(text_has_answer(ground_truth, solution_str))

    combined_score = em_reward * 10 + f1_reward * 5   # Example weights: EM is more important
    
    return combined_score


def compute_score_2wiki(config, solution_str, ground_truth, data_source, extra_info): 
    solution_str = extract_answer(solution_str)
    # print('rollout result: ', sequences_str)
    question = extra_info["question"]
    endtask_input = build_prompt_2wiki(solution_str, question)
   
    if isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.tolist()  
    elif not isinstance(ground_truth, list):
        ground_truth = ast.literal_eval(ground_truth)
    
    # reward model
    resp = call_api_reward_model(config, endtask_input, ground_truth,data_source, extra_info)
    # print('debug resp:', resp)
    endtask_llm_output = resp.json()['choices'][0]['message']['content']

    em_reward = any([exact_match(endtask_llm_output, answer) for answer in ground_truth])
    
    f1_reward = max([f1_score(endtask_llm_output, answer) for answer in ground_truth])
    
    # has_answer_reward = int(text_has_answer(ground_truth, solution_str))

    combined_score = em_reward * 10 + f1_reward * 10   # Example weights: EM is more important
    
    return combined_score


 # call prime_query_openai_async
def call_api_reward_model(config, sequences_str, ground_truth,data_source, task_extra_info) -> float:
    # print("into compute_score")
    # final_sequence_str = f"{sequences_str}\nReference:\n{ground_truth}"
    final_sequence_str = sequences_str
    return prime_query_openai_async(config, final_sequence_str)


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


def extract_answer(answer):
    if "im_end" in answer:
        answer = answer.replace("<|im_end|>", "")
    return answer.strip()


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


def build_prompt_tqa(summary, question):
    prompt = f"""
[Instruction] Answer the question. 
IMPORTANT: Respond ONLY with the exact answer in the same format as the examples. 
Do NOT add any extra text, explanations, or punctuation. Do NOT include "Answer:" or any similar prefix in your response.

[Examples]
Question: Which British politician was the first person to be made an Honorary Citizen of the United States of America?
Answer: Winston Churchill

Question: Which event of 1962 is the subject of the 2000 film Thirteen Days’, starring Kevin Costner?
Answer: The Cuban Missile Crisis

Question: Which country hosted the 1968 Summer Olympics?
Answer: Mexico

[Current Question]
{summary} 
Question: {question}
Answer:
"""
    return prompt.strip()


def build_prompt_hotpotqa(summary, question):
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


def build_prompt_2wiki(summary, question):
    prompt = f"""
[Instruction] Answer the question. 
IMPORTANT: Respond ONLY with the exact answer in the same format as the examples. 
Do NOT add any extra text, explanations, or punctuation. Do NOT include "Answer:" or any similar prefix in your response.

[Examples]
Question: Which major Russian city borders the body of water in which Saaremaa is located?
Answer: Saint Petersburg

Question: Who was thee first president of the association that wrote the code of ethics for psychology?
Answer: G. Stanley Hall

Question: Where did the Baldevins bryllup director die?
Answer: Copenhagen

[Current Question]
{summary} 
Question: {question}
Answer:
"""
    return prompt.strip()
