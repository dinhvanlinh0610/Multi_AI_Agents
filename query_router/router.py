from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from llm import GroqLLM, GeminiLLM
class RouteQuery(BaseModel):
    """
    Route a user query to the most relevant datasource
    """
    datasource: Literal["vector_store", "wiki_search"] = Field(
        ...,
        description="Given a user question choose to route it to wiki_search or vector_store"
    )

class QueryRouter():
    def __init__(self, llm, topic):

        self.llm = llm
        self.structured_router = self.llm.with_structured_output(RouteQuery)
        self.system = """You are an expert at routing a user question to a vector_store or wiki_search.
        The vector_store contains documents related to {topic}.
        Use the vector_store for questions on theess topics. Otherwise, use wiki_search.""".format(topic=topic)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system),
                ("human", "{question}"),
            ]
        )
        self.question_router = self.prompt | self.structured_router
