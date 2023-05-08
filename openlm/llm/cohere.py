

from openlm.llm.base import BaseModel
import os
from typing import Any, Dict, List, Optional, Union
import json
import requests

cohere_models = [
    'command',
    'command-nightly',
    'command-light',
    'command-light-nightly',
]

class Cohere(BaseModel):
    def __init__(self,
                 api_key = os.environ.get("COHERE_API_KEY"),
                 model_list = cohere_models,
                 namespace = 'cohere.ai',
                 base_url = 'https://api.cohere.ai/v1/generate'):
        self.api_key = api_key
        self.model_list = model_list
        self._namespace = namespace
        self.base_url = base_url

    def list_models(self):
        return self.model_list
    
    def namespace(self):
        return self._namespace
    
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
            'prompt': prompt,
            'model': model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'p': top_p,
            'frequency_penalty': frequency_penalty,
            'presence_penalty': presence_penalty,
            'stop_sequences': stop,
        }
        payload_str = json.dumps({k: v for k, v in payload.items() if v is not None})
        resp = requests.post(self.base_url, headers=headers, data=payload_str)
        if resp.status_code != 200:
            raise ValueError(resp.status_code, resp.text)
        return self._convert_response(resp.json())

    def _convert_request(req):
        return {
            'prompt': req.prompt,
            'top_p': req.top_p,
            'temperature': req.temperature,
            'max_new_tokens': req.max_tokens,
        }
    
    def _convert_response(self, resp):
        return {
            'text': resp['generations'][0]['text'],
            'extra': {
                'request_id': resp['id'],
                'generation_id': resp['generations'][0]['id'],
            }
        }
