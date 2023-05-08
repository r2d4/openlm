# OpenLM

Drop-in OpenAI-compatible library that can call LLMs from other providers (e.g., HuggingFace, Cohere, and more). 

```diff
1c1
< import openai
---
> import openlm as openai

completion = openai.Completion.create(
    model=["bloom-560m", "cohere.ai/command"], 
    prompt=["Hello world", "second prompt and then"]
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

### Other Languages
[r2d4/llm.ts](https://github.com/r2d4/llm.ts) is a TypeScript library that has a similar API that sits on top of multiple language models.

### Roadmap
- [ ] Streaming API
- [ ] Embeddings API

### Contributing
Contributions are welcome! Please open an issue or submit a PR.

### License
[MIT](LICENSE)

