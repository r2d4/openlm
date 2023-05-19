
from typing import Any, Dict, List, Optional, Union
import uuid
from openlm.llm import BaseModel, OpenAI, Huggingface, Cohere
import time
import openlm
from concurrent.futures import ThreadPoolExecutor

class Completion():
    """
    OpenAI-compatible completion API
    """
    models = {}
    aliases = {}

    @classmethod
    def create(cls, model: Union[str, List[str]], prompt: Union[str, List[str]],
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
                                user: Optional[str] = None,
                                api_keys: Optional[Dict[str, str]] = None,
                                request_timeout=0) -> Dict[str, Any]:
        """
        Creates a completion request for the OpenAI API.

        :param model: The ID(s) of the model to use.
        :param prompt: The prompt(s) to generate completions for.
        :param suffix: A string to append to the completion(s).
        :param max_tokens: The maximum number of tokens to generate in the completion(s).
        :param temperature: The sampling temperature to use.
        :param top_p: The nucleus sampling probability to use.
        :param n: The number of completions to generate.
        :param stream: Whether to stream back partial progress updates.
        :param logprobs: The number of log probabilities to generate per token.
        :param echo: Whether to include the prompt(s) in the completion(s).
        :param stop: The stop sequence(s) to use.
        :param presence_penalty: The presence penalty to use.
        :param frequency_penalty: The frequency penalty to use.
        :param best_of: The number of completions to generate and return the best of.
        :param logit_bias: A dictionary of token IDs and bias values to use.
        :param user: The ID of the user making the request.
        :return: A dictionary containing the completion response.
        """
        cls.register_default()
        if isinstance(model, str):
            model = [model]

        if isinstance(prompt, str):
            prompt = [prompt]

        # Create a list of tuples, each containing all the parameters for a call to _generate_completion
        args = [(m, p, suffix, max_tokens, temperature, top_p, n, stream, logprobs, echo, stop, presence_penalty, frequency_penalty, best_of, logit_bias, user) 
                for m in model for p in prompt]

        total_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }

        # Use a ThreadPoolExecutor to run _generate_completion in parallel for each set of parameters
        with ThreadPoolExecutor() as executor:
            choices = list(executor.map(lambda params: cls._generate_completion(*params), args))

        # Sum up the usage from all choices
        for choice in choices:
            if 'usage' in choice:
                total_usage['prompt_tokens'] += choice['usage']['prompt_tokens']
                total_usage['completion_tokens'] += choice['usage']['completion_tokens']
                total_usage['total_tokens'] += choice['usage']['total_tokens']

        return {
            "id": str(uuid.uuid4()),
            "object": "text_completion",
            "created": int(time.time()),
            "choices": choices,
            "usage": total_usage,
        }
    
    @classmethod
    def _generate_completion(cls, model, prompt, suffix, max_tokens, temperature, top_p, n, stream, logprobs, echo, stop, presence_penalty, frequency_penalty, best_of, logit_bias, user):
        """
        Function to generate a single completion. This will be used in parallel execution.
        """
        if model not in cls.aliases:
            raise ValueError(f"Model {model} not found. OpenLM currently supports the following models:\n{cls._pretty_list_models()}")
        fqn = cls.aliases[model]
        try:
            ret = cls.models[fqn].create_completion(
                model=fqn[len(cls.models[fqn].namespace())+1:],
                prompt=prompt, 
                suffix=suffix, 
                max_tokens=max_tokens, 
                temperature=temperature, 
                top_p=top_p, 
                n=n, 
                stream=stream, 
                logprobs=logprobs, 
                echo=echo, 
                stop=stop, 
                presence_penalty=presence_penalty, 
                frequency_penalty=frequency_penalty, 
                best_of=best_of, 
                logit_bias=logit_bias, 
                user=user)
        except Exception as e:
            ret = {
                'error': f"Error: {e}"
            }
        choice = {
            "id": str(uuid.uuid4()),
            "model_name": fqn,
            'created': int(time.time()),
        }
        if 'error' in ret:
            choice['error'] = ret['error']
        if 'text' in ret:
            choice['text'] = ret['text']
        if 'usage' in ret:
            choice['usage'] = ret['usage']
        if 'extra' in ret:
            choice['extra'] = ret['extra']
        return choice
    
    @classmethod
    def register(cls, providers: BaseModel | List[BaseModel]):
        if not isinstance(providers, list):
            providers = [providers]
        for provider in providers:
            for model in provider.list_models():
                fqn = provider.namespace() + '/' + model
                cls.models[fqn] = provider
                cls.aliases[model] = fqn
                cls.aliases[fqn] = fqn
                if '/' in model:
                    cls.aliases[model.split('/')[1]] = fqn

    @classmethod
    def register_default(cls, api_keys: Optional[Dict[str, str]] = None):
        if openlm.api_key:
            cls.register(OpenAI(api_key=openlm.api_key))
        else:
            if api_keys and api_keys['openai.com'] is not None:
                cls.register(OpenAI(api_key=api_keys['openai.com']))
            else:
                cls.register(OpenAI())
        if api_keys and api_keys['huggingface.co'] is not None:
            cls.register(Huggingface(api_key=api_keys['huggingface.co']))
        else:
            cls.register(Huggingface())
        if api_keys and api_keys['cohere.ai'] is not None:
            cls.register(Cohere(api_key=api_keys['cohere.ai']))
        else:
            cls.register(Cohere())

    @classmethod
    def list_models(cls) -> List[str]:
        reverse_alias = {}
        for key, value in cls.aliases.items():
            # If the value is not in the reverse dictionary, create an empty array for it
            if value not in reverse_alias:
                reverse_alias[value] = []
            # Append the key to the array for the value in the reverse dictionary
            reverse_alias[value].append(key)

        return reverse_alias
    
    @classmethod
    def _pretty_list_models(cls):
        ret = ""
        for key, value in cls.list_models().items():
            ret += f"-> {value} \n"
        return ret


