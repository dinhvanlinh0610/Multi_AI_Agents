from langchain_experimental.text_splitter import SemanticChunker

class SemanticSplitter():
    def __init__(self, embedding):
        """
        Initialize the SemanticSplitter

        Args:
            embedding (object): Embedding object
            
        """
        self.embedding_model = embedding.embeddings
        self.text_splitter = SemanticChunker(embeddings=self.embedding_model, breakpoint_threshold_type="percentile") 
        self.docs = []

    def splits(self, documents):
        """
        Split the documents into chunks

        Args:
            documents (list): List of Document objects

        Returns:
            list: List of documents

        """
        self.docs = self.text_splitter.split_documents(documents)
        return self.docs