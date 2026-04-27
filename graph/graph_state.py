from typing import TypedDict
from dataclasses import dataclass, field

from langchain.chat_models import BaseChatModel


@dataclass
class GraphState:
    file_path: str = field(default_factory=str)
    aim_path: str = field(default_factory=str)


class GraphContext(TypedDict):
    model: BaseChatModel