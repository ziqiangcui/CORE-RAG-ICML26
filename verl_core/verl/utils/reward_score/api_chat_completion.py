# -*- coding: utf-8 -*-
import json
import uuid
import requests
from typing import Dict, Optional, Generator
import aiohttp

class OpenAPI:
    def __init__(
            self,
            base_url: str, 
            model: str,    
            stream: bool = False,
            temperature: float = 0.7,
            top_p: float = 0.6,
            top_k: int = 20,
            repetition_penalty: float = 1.05,
            output_seq_len: int = 2048,
            max_input_seq_len: int = 2048
    ):
        self.base_url = base_url
        self.model = model
        self.stream = stream
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.output_seq_len = output_seq_len
        self.max_input_seq_len = max_input_seq_len

    def _generate_request_id(self) -> str:
        return f"req_{uuid.uuid4().hex[:8]}"


    def _process_stream_response(self, response: requests.Response) -> Generator[str, None, None]:
        for line in response.iter_lines():
            if not line:
                continue

            try:
                line_str = line.decode('utf-8')
                if line_str.startswith('data:'):
                    data = json.loads(line_str[5:])
                    content = data.get('choices', [{}])[0].get('delta', {}).get('content')
                    if content:
                        yield content
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e} | 原始数据: {line_str}")
            except Exception as e:
                print(f"处理流数据时发生错误: {e}")

    def _process_non_stream_response(self, response: requests.Response) -> str:
        """处理非流式响应"""
        try:
            return response.json()['choices'][0]['message']['content']
        except KeyError:
            raise ValueError("响应格式异常，缺少必要字段")
        except json.JSONDecodeError:
            raise ValueError("无效的JSON响应")

    def chat_completion(
            self,
            messages: list,
            **kwargs
    ) -> Generator[str, None, None]:
        """
        执行聊天补全请求

        :param messages: 消息列表
        :param model: 使用的模型名称
        :param stream: 是否启用流式模式
        :param kwargs: 其他请求参数（temperature, top_p等）
        :return: 生成器（流式模式）或完整响应字符串
        """
        # 平台api
        request_data = {
            "model": self.model,
            "user": "abc",
            "query_id": self._generate_request_id(),
            "messages": messages,
            "stream": self.stream,
            "do_sample": False,
            "maxTokens": 8192,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "output_seq_len": self.output_seq_len,
            "max_input_seq_len": self.max_input_seq_len,
            **kwargs
        }
        try:
            response = requests.post(
                url=self.base_url,
                # headers=self._build_headers(),
                json=request_data,
                stream=self.stream,
                timeout=30
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"API请求失败: {str(e)}")

        return response


# 使用示例
if __name__ == '__main__':
    client = OpenAPI()

    # example
    messages = [
        {'role': 'system', 'content': ''},
        {'role': 'user', 'content': 'hello'}
    ]

    result = client.chat_completion(messages=messages)
    print(result.json())
