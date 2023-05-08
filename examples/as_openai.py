import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import openlm as openai
import json

completion = openai.Completion.create(
    model="ada", 
    prompt="Hello world"
)
print(json.dumps(completion, indent=4))

'''
{
    "id": "3890d5c3-e6c4-4222-b77d-40a65f1d032b",
    "object": "text_completion",
    "created": 1683583320,
    "choices": [
        {
            "id": "660d576d-8c04-420f-b410-146729c8fc8a",
            "model_idx": 0,
            "model_name": "openai.com/ada",
            "index": 0,
            "created": 1683583320,
            "text": ". Details?), website modding, something that makes no sense, and just design",
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 16,
                "total_tokens": 18
            },
            "extra": {
                "id": "cmpl-7E3DcN1o6Axv1JIXxHNt11Q0wA1Is"
            }
        }
    ],
    "usage": {
        "prompt_tokens": 2,
        "completion_tokens": 16,
        "total_tokens": 18
    }
}
'''