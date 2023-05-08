import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import openlm
import json
from typing import Any, Dict, List, Optional, Union

class CustomModel(openlm.BaseModel):
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
        # completions should return a dictionary with the following keys:
        return {
            # Required keys:
            "text": "Hello world!"

            ## Optional keys:
            # ,'extra': {
            #    'key': 'value'
            # },
            # 'usage': {
            #   'prompt_tokens': 0,
            #   'completion_tokens': 0,
            #   'total_tokens': 0,
            # }
        }
    
    def list_models(self) -> Dict[str, Any]:
        # list of model names that can be used with this provider
        return ["your_model_name"]
    
    def namespace(self) -> str:
        # A namespace prevents name collisions between models from different providers.
        # You will be able to reference your model both as:
        # your_namespace/your_model_name or your_model_name
        return "your_namespace"
    
openlm.Completion.register(CustomModel())

# Now you can use your custom model in the same way as the built-in models:
completion = openlm.Completion.create(
    model="your_model_name",
    prompt="Hello world"
)

print(json.dumps(completion, indent=4))

'''
{
    "id": "12bf5515-e2cc-463d-b120-c21c911364f9",
    "object": "text_completion",
    "created": 1683583298,
    "choices": [
        {
            "id": "2dde9e4e-17c3-4d92-be6f-285fb9a96935",
            "model_idx": 0,
            "model_name": "your_namespace/your_model_name",
            "index": 0,
            "created": 1683583298,
            "text": "Hello world!"
        }
    ],
    "usage": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }
}
'''