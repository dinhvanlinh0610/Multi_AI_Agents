from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

class RouteQuery(BaseModel):
    """
    Route a user query to the most relevant datasource
    """
    datasource: Literal["vector_store", "wiki_search"] = Field(
        ...,
        description="Given a user question choose to route it to wiki_search or vector_store"
    )