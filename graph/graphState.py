from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document



class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

