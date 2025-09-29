# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import asyncio
import torch
from typing import List, Any, Optional
from transformers import AutoTokenizer
try:
    from .api_chat_completion import OpenAPI
except ImportError:
    from api_chat_completion import OpenAPI
import json
from typing import Dict, Generator
import requests


def extract_output(solution_text: str):
    # Match everything inside the last \boxed{} in the solution text
    boxed_pattern = r'\\bold{(.*)}'
    matches = re.findall(boxed_pattern, solution_text)
    if matches:
        return matches[-1].strip()
    return None


def process_stream_response(self, response: requests.Response) -> Generator[str, None, None]:
    for line in response.iter_lines():
        if not line:
            continue

        try:
            line_str = line.decode('utf-8')
            if line_str.startswith('data:'):
                data = json.loads(line_str[5:])
                content = data.get('choices', [{}])[0].get('delta', {}).get('content').strip()
                if content:
                    yield content
        except json.JSONDecodeError as e:
            print(f"JSON error: {e} | original data: {line_str}")
        except Exception as e:
            print(f"error: {e}")


def process_non_stream_response(self, response: requests.Response) -> str:
    try:
        return response.json()['choices'][0]['message']['content'].strip()
    except KeyError:
        raise ValueError("error")
    except json.JSONDecodeError:
        raise ValueError("error")


def prime_query_openai_async(config, sequence_str: str) -> Optional[requests.Response]:
    api_client = OpenAPI(base_url=config['url'], model=config['model'], stream=config['stream'],
                              temperature=config['temperature'],
                              top_p=config['top_p'], top_k=config['top_k'],
                              repetition_penalty=config['repetition_penalty'],
                              output_seq_len=config['output_seq_len'], max_input_seq_len=config['max_input_seq_len'])
    max_retries = config['max_retries']
    for attempt in range(max_retries):
        # print('into while messages , attempt: ', attempt)
        try:
            messages = [
                {'role': 'system', 'content': config['system_prompt']},
                {'role': 'user', 
                 'content': sequence_str
                #  'content': config['scoring_prompt'] + '\n' + sequence_str
                 }
            ]
            # if using vllm server
            if config['tokenizer'] and config['apply_chat_template']:
                tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
                messages = tokenizer['apply_chat_template'](messages, tokenize=False, add_generation_prompt=True)
            # print('into client chat completion')
            response = api_client.chat_completion(messages=messages)

            return response
     
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                return None
    return None


async def _query_openai_async(client: OpenAPI, sequence_str: str, config, semaphore: asyncio.Semaphore,
                              index: int) -> tuple[int, float]:
    """
    Query OpenAI API asynchronously.
    """
    max_retries = config['max_retries']
    # scoring_prompt = config['scoring_prompt']
    async with semaphore:
        for attempt in range(max_retries):
            print('into while messages , attempt: ',attempt)
            try:
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": sequence_str
                        # "content": scoring_prompt + '\n' + sequence_str
                    },
                ]
                # if using vllm server
                if config['tokenizer'] and config['apply_chat_template']:
                    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
                    messages = tokenizer['apply_chat_template'](messages, tokenize=False, add_generation_prompt=True)
                print('into client chat completion')
                response = await asyncio.to_thread(client.chat_completion, messages=messages)
                # response.json()['choices'][0]['message']['content']
                print('respense is ok!')
                content = response.json()['choices'][0]['message']['content']
                print('content info: ', content)
                extracted = extract_output(content)
                print('extracted is : ', extracted)
                if extracted is not None:
                    return index, float(extracted)
                else:
                    raise ValueError("No valid score extracted")
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return index, config['default_score']
                await asyncio.sleep(2 ** attempt)
        return index, config['default_score']


async def process_data_async(data_source: List[str], solution_str: List[str], ground_truth: List[str],
                             extra_info: List[Dict[str, Any]], config) -> torch.Tensor:
    """
    Process data asynchronously using API.
    """
    reward_tensor = torch.zeros(len(solution_str), dtype=torch.float32)
    api_client = OpenAPI(base_url=config['url'], model=config['model'], stream=config['stream'],
                              temperature=config['temperature'],
                              top_p=config['top_p'], top_k=config['top_k'],
                              repetition_penalty=config['repetition_penalty'],
                              output_seq_len=config['output_seq_len'], max_input_seq_len=config['max_input_seq_len'])
    semaphore = asyncio.Semaphore(config['max_concurrency'])

    tasks = []
    for i in range(len(solution_str)):
        prompt = solution_str[i]
        response = ground_truth[i]
        if response is None:
            sequence_str = prompt
        else:
            sequence_str = f"{prompt}\nReference:\n{response}"

        task = asyncio.create_task(_query_openai_async(api_client, sequence_str, config, semaphore, i))
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    for i, score in results:
        print('i :', i)
        print('score', score)
        reward_tensor[i] = score

    return reward_tensor

