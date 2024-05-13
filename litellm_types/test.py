import asyncio
from .completion import AsyncStream
from .messages import AssistantMessage, UserMessage, AnyMessage
from pydantic import BaseModel
from litellm import completion
from typing import Literal
from rich import print


class CheckWeather(BaseModel):
    location: str


# class TestUserMesage(BaseModel):
#     content: str
#     role: Literal["user", "assistant"]


# ub = completion(
#     model="gpt-3.5-turbo",
#     messages=[
#         UserMessage(content="Check the weather in paris."),
#     ],
# )
# print(ub)


async def main():
    # chat: list[AnyMessage] = [
    #     UserMessage(content="Check the weather in paris."),
    # ]
    # print(chat)
    # stream_gen = AsyncStream(model="gpt-3.5-turbo", tools=[CheckWeather])
    # stream = await stream_gen(chat)
    # full_msg = AssistantMessage(content="")
    # async for message in stream:
    #     full_msg += message
    # print(full_msg)

    chat: list[AnyMessage] = [
        UserMessage(content="Check the weather in paris."),
    ]
    print(chat)
    stream_gen = AsyncStream(model="gpt-3.5-turbo")  # , tools=[CheckWeather])
    stream = await stream_gen(chat)
    full_msg = AssistantMessage(content="")
    async for message in stream:
        full_msg += message
    print(full_msg)


if __name__ == "__main__":
    asyncio.run(main())
