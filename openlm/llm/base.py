import abc
from typing import Any, Dict, List, Optional, Union


class BaseCompletion(metaclass=abc.ABCMeta):
    @abc.abstractmethod
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
        raise NotImplementedError


class BaseModel(BaseCompletion, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def list_models(self) -> Dict[str, Any]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def namespace(self) -> str:
        raise NotImplementedError   