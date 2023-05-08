import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import openlm
import json

completion = openlm.Completion.create(
    model=["ada", "distilgpt2", "huggingface.co/"], 
    prompt="Hello world",
    api_keys={
        'huggingface.co': 'YOUR_API_KEY', # or os.environ["HF_API_TOKEN"]
        'cohere.ai': 'YOUR_API_KEY', # or os.environ["COHERE_API_KEY"]
        'openai.com': 'YOUR_API_KEY' # or os.environ["OPENAI_API_KEY"]
    },
)

print(json.dumps(completion, indent=4))
