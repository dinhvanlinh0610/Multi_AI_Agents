from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class RSTextSplitter():
    def __init__(self, text: str = None, documents : Document = None):
        """
        Initialize the RSTextSplitter

        Args:
            text (str, optional): Text to split. Defaults to None.
            documents (Document, optional): Document object. Defaults to None.

        """
        self.text = text
        self.documents = documents
        self.docs = []

    def split_documents(self, chunk_size, chunk_overlap):
        """
        Split the documents into chunks

        Args:
            chunk_size (int): Size of the chunk
            chunk_overlap (int): Overlap between the chunks

        Returns:
            list: List of documents
        """
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.docs = text_splitter.split_documents(self.documents)
        return self.docs
