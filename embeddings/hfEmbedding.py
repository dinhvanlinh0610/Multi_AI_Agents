from langchain_huggingface import HuggingFaceEmbeddings

class HFEmbedding():
    def __init__(self, model_name):
        """
        Initialize the HFEmbedding

        Args:
            model_name (str): Model name

        """
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

    def get_embedding(self, text):
        """
        Get the embedding of the text

        Args:
            text (str): Text to embed
            
        """
        embedding = self.embeddings.embed_query(text)
        return embedding
