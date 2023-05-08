import json
import os
from typing import Any, Dict, List, Optional, Union

import requests

from openlm.llm.base import BaseModel

openai_models = [
            'text-davinci-003', 
            'text-davinci-002', 
            'text-curie-001', 
            'text-babbage-001', 
            'text-ada-001',
            
            # aliases
            'ada',
            'babbage',
            'curie',
            'davinci',
        ]

class OpenAI(BaseModel):
    def __init__(self,
                 api_key = os.environ.get("OPENAI_API_KEY"), 
                 model_list = openai_models, 
                 namespace = 'openai.com', 
                 base_url = 'https://api.openai.com/v1/completions'):
        
        if api_key is None:
            raise ValueError("OPENAI_API_KEY is not set or passed as an argument")
        
        self.api_key = api_key
        self.model_list = model_list
        self._namespace = namespace
        self.base_url = base_url

    def create_completion(self, model: Union[str, List[str]], prompt: Union[str, List[str]],
                                suffix: Optional[str] = None,
                                max_tokens: Optional[int] = None,
                                temperature: Optional[float] = None,
                                top_p: Optional[float] = None,
                                n: Optional[int] = None,
                                stream: Optional[bool] = None,
                                logprobs: Optional[int] = None,
                                echo: Optional[bool] = None,
                                stop: Optional[Union[str, List[str]]] = None,
                                presence_penalty: Optional[float] = None,
                                frequency_penalty: Optional[float] = None,
                                best_of: Optional[int] = None,
                                logit_bias: Optional[Dict[str, float]] = None,
                                user: Optional[str] = None) -> Dict[str, Any]:
        headers = {'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'}
        payload = {
            'model': model,
            'prompt': prompt,
            'suffix': suffix,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'n': n,
            'stream': stream,
            'logprobs': logprobs,
            'echo': echo,
            'stop': stop,
            'presence_penalty': presence_penalty,
            'frequency_penalty': frequency_penalty,
            'best_of': best_of,
            'logit_bias': logit_bias,
            'user': user
        }

        payload_str = json.dumps({k: v for k, v in payload.items() if v is not None})
        resp = requests.post(self.base_url, headers=headers, data=payload_str).json()
        if 'error' in resp:
            raise ValueError(resp['error'])
        return self._convert_response(resp)

    def _convert_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'text': response['choices'][0]['text'],
            'extra': {
                'id': response['id'],
            },
            'usage': response['usage'],
        }

    def list_models(self):
        return self.model_list
    
    def namespace(self):
        return self._namespace