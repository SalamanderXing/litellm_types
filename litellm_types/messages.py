from pydantic import BaseModel
from typing import Any, Literal, get_type_hints, Awaitable
from beartype import beartype
from typing import Callable, TypeVar
from functools import wraps, update_wrapper
import functools
import httpx
import inspect
from litellm import acompletion_with_retries
from litellm.utils import Function, ChatCompletionDeltaToolCall
from rich import print


# class StrictDict(dict):
#     def __getattr__(self, item):
#         if item in self:
#             return self[item]
#         raise AttributeError(f"'UserMessage' object has no attribute '{item}'")

#     def __setattr__(self, key, value):
#         if key in self:
#             self[key] = value
#         else:
#             raise AttributeError(f"Cannot set unknown attribute '{key}'")


class FunctionMessage(BaseModel):
    content: str
    name: str
    role: Literal["function"] = "function"

    def __init__(self, content: str, name: str):
        super().__init__(content=content, name=name, role="function")


class UserMessage(BaseModel):
    content: str
    role: Literal["user"] = "user"

    def __init__(self, content: str):
        super().__init__(content=content, role="user")


class SystemMessage(BaseModel):
    content: str
    role: Literal["system"] = "system"

    def __init__(self, content: str):
        super().__init__(content=content, role="system")


class ToolCall(BaseModel):
    id: str
    function: Function
    type: str

    @beartype
    def __init__(self, id: str, function: Function):
        super().__init__(id=id, function=function, type="function")


class ToolCallDelta(BaseModel):
    id: str | None = None
    function: Function
    type: str | None = None


def add_tool_calls(
    a: ToolCallDelta | ToolCall, b: ToolCallDelta | ToolCall
) -> ToolCallDelta:
    res_fn = {
        k: best_of_both(a.function.__dict__[k], b.function.__dict__[k])
        for k in a.function.__dict__
    }
    fn = Function(**res_fn)  # type: ignore
    return ToolCallDelta(
        id=best_of_both(a.id, b.id),
        function=fn,
        type=best_of_both(a.type, b.type),
    )


class AssistantMessageDelta(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ChatCompletionDeltaToolCall] | None = None


def best_of_both(a, b):
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return a + b


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str
    tool_calls: list[ToolCall | ToolCallDelta] | None = None

    def __init__(
        self, content: str, tool_calls: list[ToolCall | ToolCallDelta] | None = None
    ):
        super().__init__(content=content, role="assistant", tool_calls=tool_calls)

    def __add__(
        self, other: "AssistantMessage | AssistantMessageDelta"
    ) -> "AssistantMessage":
        res_tool_calls = []
        if not self.tool_calls is None:
            res_tool_calls = self.tool_calls
        if not other.tool_calls is None:
            res_tool_calls = other.tool_calls

        if self.tool_calls is not None and other.tool_calls is not None:
            # import ipdb

            # ipdb.set_trace()
            res_tool_calls = [
                add_tool_calls(t1, t2)
                for t1, t2 in zip(self.tool_calls, other.tool_calls)
            ]

        # import ipdb

        # ipdb.set_trace()
        return AssistantMessage(
            content=self.content + (other.content if other.content else ""),
            tool_calls=res_tool_calls,
        )


class ToolMessage(BaseModel):
    @beartype
    def __init__(self, content: str, tool_call_id: str):
        super().__init__(content=content, role="tool", tool_call_id=tool_call_id)


AnyMessage = UserMessage | AssistantMessage | SystemMessage | FunctionMessage
