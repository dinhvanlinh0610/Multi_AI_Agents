from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class RSTextSplitter():
    def __init__(self, text: str = None, documents : Document = None):
        self.text = text
        self.documents = documents
        self.docs = []

    def split_documents(self, chunk_size, chunk_overlap):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.docs = text_splitter.split_documents(self.documents)
        return self.docs
