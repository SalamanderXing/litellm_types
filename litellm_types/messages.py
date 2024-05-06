from pydantic import BaseModel
from typing import Any, Literal, get_type_hints, Awaitable
from beartype import beartype
from typing import Callable, TypeVar
from functools import wraps, update_wrapper
import functools
import httpx
import inspect
from litellm import acompletion_with_retries


class StrictDict(dict):
    def __getattr__(self, item):
        if item in self:
            return self[item]
        raise AttributeError(f"'UserMessage' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        if key in self:
            self[key] = value
        else:
            raise AttributeError(f"Cannot set unknown attribute '{key}'")


class FunctionMessage(StrictDict):
    @beartype
    def __init__(self, content: str, name: str):
        super().__init__(role="tool", content=content, name=name)


class UserMessage(StrictDict):
    @beartype
    def __init__(self, content: str):
        super().__init__(role="user", content=content)


class SystemMessage(dict):
    @beartype
    def __init__(self, content: str):
        super().__init__(role="system", content=content)


class AssistantMessage(StrictDict):
    @beartype
    def __init__(self, content: str):
        super().__init__(role="assistant", content=content)


AnyMessage = UserMessage | AssistantMessage | SystemMessage | FunctionMessage
