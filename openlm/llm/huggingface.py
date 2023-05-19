

from openlm.llm.base import BaseModel
import os
from typing import Any, Dict, List, Optional, Union
import json
import requests

hf_models = [
    'gpt2',
    'distilgpt2',
    'gpt2-large',
    'gpt2-medium',
    'gpt2-xl',

    'bigscience/bloom-560m',
    'bigscience/bloom-1b',
    'bigscience/bloom-3b',
    'bigscience/bloom-7b1',

    'decapoda-research/llama-7b-hf',
    'decapoda-research/llama-13b-hf',
    'decapoda-research/llama-30b-hf',
    'decapoda-research/llama-65b-hf',
    
    'EleutherAI/gpt-j-6B',
    'EleutherAI/gpt-j-2.7B',

    'EleutherAI/gpt-neo-125M',
    'EleutherAI/gpt-neo-1.3B',
    'EleutherAI/gpt-neox-20B',

    'EleutherAI/pythia-160m',
    'EleutherAI/pythia-70m',
    'EleutherAI/pythia-12b',
    
    'cerebras/Cerebras-GPT-111M',
    'cerebras/Cerebras-GPT-1.3B',
    'cerebras/Cerebras-GPT-2.7B',
    
    'bigcode/santacoder',
    
    'Salesforce/codegen-350M-multi',
    'Salesforce/codegen-2b-multi',
    
    'stabilityai/stablelm-tuned-alpha-3b',
    'stabilityai/stablelm-tuned-alpha-7b',

    'facebook/opt-125m',
    'facebook/opt-350m',
    'facebook/opt-1.3b',
    'facebook/opt-2.7b',
    'facebook/opt-6.7b',
    'facebook/opt-13b',
    'facebook/opt-30b',

    'mosaicml/mpt-7b',
    'mosaicml/mpt-7b-instruct',
    
    'databricks/dolly-v2-7b',
    'databricks/dolly-v2-12b',


]

class Huggingface(BaseModel):
    def __init__(self,
                 api_key = os.environ.get("HF_API_TOKEN"),
                 model_list = hf_models,
                 namespace = 'huggingface.co',
                 base_url = 'https://api-inference.huggingface.co/models'):
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
            'inputs': prompt,
            'top_p': top_p,
            'temperature': temperature,
            'max_new_tokens': max_tokens,
        }
        payload_str = json.dumps({k: v for k, v in payload.items() if v is not None})
        resp = requests.post(self.base_url + '/' + model, headers=headers, data=payload_str)
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
            'text': resp[0]['generated_text'],
        }
