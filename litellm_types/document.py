from pydantic import BaseModel
from typing import Any


class Document(BaseModel):
    page_content: str
    metadata: dict[str, Any]
