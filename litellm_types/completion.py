from pydantic import BaseModel
from typing import Any, Literal, get_type_hints, Awaitable
from beartype import beartype
from typing import Callable, TypeVar
from functools import wraps, update_wrapper
import functools
import httpx
import inspect

#  from litellm import acompletion_with_retries
from litellm import acompletion
from openai import AsyncClient
from rich import print
from .messages import AssistantMessage, AnyMessage


ACompletion = Callable[[list[AnyMessage]], Awaitable[AssistantMessage]]


def get_acompletion(
    model: str,
    # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
    timeout: float | str | httpx.Timeout | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    n: int | None = None,
    stream: bool | None = None,
    stop=None,
    max_tokens: int | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    logit_bias: dict | None = None,
    user: str | None = None,
    # openai v1.0+ new params
    response_format: dict | None = None,
    seed: int | None = None,
    tools: list | None = None,
    tool_choice: str | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    deployment_id=None,
    extra_headers: dict | None = None,
    # soon to be deprecated params by OpenAI
    functions: list | None = None,
    function_call: str | None = None,
    # set api_base, api_version, api_key
    base_url: str | None = None,
    api_version: str | None = None,
    api_key: str | None = None,
    model_list: list | None = None,  # pass in a list of api_base,keys, etc.
    # Optional liteLLM function params
    **kwargs,
) -> ACompletion:

    async def wrapper(messages: list[AnyMessage]) -> AssistantMessage:
        result = await acompletion(
            model=model,
            messages=messages,
            # timeout=timeout,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stop=stop,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            user=user,
            response_format=response_format,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            deployment_id=deployment_id,
            extra_headers=extra_headers,
            functions=functions,
            function_call=function_call,
            base_url=base_url,
            api_version=api_version,
            api_key=api_key,
            model_list=model_list,
            **kwargs,
        )
        print(result)
        result = result.choices[0].message.content
        return AssistantMessage(result)

    return wrapper


def get_acompletion_from_openai(
    client: AsyncClient,
    model: str,
    # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
    timeout: float | str | httpx.Timeout | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    n: int | None = None,
    stream: bool | None = None,
    stop=None,
    max_tokens: int | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    logit_bias: dict | None = None,
    user: str | None = None,
    # openai v1.0+ new params
    response_format: dict | None = None,
    seed: int | None = None,
    tools: list | None = None,
    tool_choice: str | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    deployment_id=None,
    extra_headers: dict | None = None,
    # soon to be deprecated params by OpenAI
    functions: list | None = None,
    function_call: str | None = None,
    # set api_base, api_version, api_key
    base_url: str | None = None,
    api_version: str | None = None,
    api_key: str | None = None,
    model_list: list | None = None,  # pass in a list of api_base,keys, etc.
    # Optional liteLLM function params
    **kwargs,
) -> ACompletion:

    async def wrapper(messages: list[AnyMessage]) -> AssistantMessage:
        result = await client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=timeout,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stop=stop,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            functions=functions,
            api_key=api_key,
            **kwargs,
        )
        result = result.choices[0].message.content
        return AssistantMessage(result)

    return wrapper
