from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=256)

wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
