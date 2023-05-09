# OpenLM

Drop-in OpenAI-compatible library that can call LLMs from other providers (e.g., HuggingFace, Cohere, and more). 

```diff
1c1
< import openai
---
> import openlm as openai

completion = openai.Completion.create(
    model=["bloom-560m", "cohere.ai/command"], 
    prompt=["Hello world!", "A second prompt!"]
)
print(completion)
```
### Features
* Takes in the same parameters as OpenAI's Completion API and returns a similarly structured response. 
* Call models from HuggingFace's inference endpoint API, Cohere.ai, OpenAI, or your custom implementation. 
* Complete multiple prompts on multiple models in the same request. 
* Very small footprint: OpenLM calls the inference APIs directly rather than using multiple SDKs.


### Installation
```bash
pip install openlm
```

### Examples

- [Import as OpenAI](examples/as_openai.py)
- [Set up API keys via environment variables or pass a dict](examples/api_keys.py)
- [Add a custom model or provider](examples/custom_provider.py)
- [Complete multiple prompts on multiple models](examples/multiplex.py)

OpenLM currently supports the Completion endpoint, but over time will support more standardized endpoints that make sense. 

### [Example with Response](examples/multiplex.py)

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import openlm 
import json

completion = openlm.Completion.create(
    model=["ada", "huggingface.co/gpt2", "cohere.ai/command"],
    prompt=["The quick brown fox", "Who jumped over the lazy dog?"],
    max_tokens=15
)
print(json.dumps(completion, indent=4))
```

```json
{
    "id": "504cc502-dc27-43e7-bcc3-b62e178c247e",
    "object": "text_completion",
    "created": 1683583267,
    "choices": [
        {
            "id": "c0487ba2-935d-4dec-b191-f7eff962f117",
            "model_idx": 0,
            "model_name": "openai.com/ada",
            "index": 0,
            "created": 1683583233,
            "text": " jumps into the much bigger brown bush.\" \"Alright, people like you can",
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 15,
                "total_tokens": 19
            },
            "extra": {
                "id": "cmpl-7E3CCSpJHXfx5yB0TaJU9ON7rNYPT"
            }
        },
        {
            "id": "bab92d11-5ba6-4da2-acca-1f3398a78c3e",
            "model_idx": 0,
            "model_name": "openai.com/ada",
            "index": 1,
            "created": 1683583233,
            "text": "\n\nIt turns out that saying one's name \"Joe\" is the",
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 15,
                "total_tokens": 22
            },
            "extra": {
                "id": "cmpl-7E3CDBbqFy92I2ZbSGoDT5ickAiPD"
            }
        },
        {
            "id": "be870636-9d9e-4f74-b8bd-d04766072a7b",
            "model_idx": 1,
            "model_name": "huggingface.co/gpt2",
            "index": 0,
            "created": 1683583234,
            "text": "The quick brown foxes, and the short, snuggly fox-scented, soft foxes we have in our household\u2026 all come in two distinct flavours: yellow and orange; and red and white. This mixture is often confused with"
        },
        {
            "id": "c1abf535-54a9-4b72-8681-d3b4a601da88",
            "model_idx": 1,
            "model_name": "huggingface.co/gpt2",
            "index": 1,
            "created": 1683583266,
            "text": "Who jumped over the lazy dog? He probably got it, but there's only so much you do when you lose one.\n\nBut I will say for a moment that there's no way this guy might have picked a fight with Donald Trump."
        },
        {
            "id": "08e8c351-236a-4497-98f3-488cdc0b6b6a",
            "model_idx": 2,
            "model_name": "cohere.ai/command",
            "index": 0,
            "created": 1683583267,
            "text": "\njumps over the lazy dog.",
            "extra": {
                "request_id": "0bbb28c0-eb3d-4614-b4d9-1eca88c361ca",
                "generation_id": "5288dd6f-3ecf-475b-b909-0b226be6a193"
            }
        },
        {
            "id": "49ce51e6-9a18-4093-957f-54a1557c8829",
            "model_idx": 2,
            "model_name": "cohere.ai/command",
            "index": 1,
            "created": 1683583267,
            "text": "\nThe quick brown fox.",
            "extra": {
                "request_id": "ab5d5e03-22a1-42cd-85b2-9b9704c79304",
                "generation_id": "60493966-abf6-483c-9c47-2ea5c5eeb855"
            }
        }
    ],
    "usage": {
        "prompt_tokens": 11,
        "completion_tokens": 30,
        "total_tokens": 41
    }
}
```

### Other Languages
[r2d4/llm.ts](https://github.com/r2d4/llm.ts) is a TypeScript library that has a similar API that sits on top of multiple language models.

### Roadmap
- [ ] Streaming API
- [ ] Embeddings API

### Contributing
Contributions are welcome! Please open an issue or submit a PR.

### License
[MIT](LICENSE)

