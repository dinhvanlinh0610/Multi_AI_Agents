from langchain_chroma import Chroma
from uuid import uuid4
class ChromaVectorStore():

    def __init__(self, collection_name = "example_collection", embeddings = None):
        """
        Initialize the ChromaVectorStore

        Args:
            collection_name (str, optional): Name of the collection. Defaults to "example_collection".
            embeddings (Embeddings, optional): Embeddings object (have function embed_documents). Defaults to None.
        """
        self.collection_name = collection_name
        self.embedding_function = embeddings.embeddings
        self.persist_directory = "./data/chroma_db"
        self.chroma = Chroma(self.collection_name, 
                             self.embedding_function, 
                             self.persist_directory)
    
    def create_retriever(self):
        """
        Create a retriever object

        Returns:
            Retriever: Retriever object
        """
        retriever = self.chroma.as_retriever(
            search_type = "mmr",
            search_kwargs = {"k": 3, "fetch_k": 5}
        )
        return retriever
        
    def add(self, documents):
        """
        Add documents to the vector store

        Args:
            documents (list): List of Document objects

        Returns:
            ChromaVectorStore: Returns the ChromaVectorStore

        """
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.chroma.add_documents(documents=documents, ids=uuids)
        return self.chroma
    
    def query_directly(self, query, k, direct):
        """
        Query the vector store with a query and return the results

        Args:
            query (str): Query string
            k (int): Number of results to return
            direct (str): Direct string

        Returns:
            list: List of results
        """
        results = self.chroma.similarity_search(
            query=query,
            k=k,
            filter={"source": direct}
        )
        return results
    def query_with_score(self, query, k):
        """
        Query the vector store with a query and return the results

        Args:
            query (str): Query string
            k (int): Number of results to return

        Returns:
            list: List of results
        """
        results = self.chroma.similarity_search_with_score(
            query=query,
            k=k,

        )
        return results
    def query_by_vector(self, query, k):
        """
        Query the vector store with a query and return the results

        Args:
            query (str): Query string
            k (int): Number of results to return

        Returns:
            list: List of results
        """
        results = self.chroma.similarity_search_by_vector(
            embedding=self.embedding_function.embed_query(query),
            k=k
        )
        return results
    