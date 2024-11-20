# from embeddings.hfEmbedding import HFEmbedding
# from vector_store.chroma_db import ChromaVectorStore
# from langchain.schema import Document
# from document_loader.webLoader import WebLoader
# from document_splitter import RSTextSplitter, SemanticSplitter
# embedding_model = HFEmbedding("all-mpnet-base-v2")

# vector_store = ChromaVectorStore(collection_name="ex3",embeddings=embedding_model.embeddings)

# # Add documents  to the vector store
# # documents = [Document(page_content = "This is a test 2 document", metadata={"source": "ex1"}),
# #              Document(page_content = "This is another test document", metadata={"source": "ex1"}),
# #              Document(page_content = "This is a third test document", metadata={"source": "ex1"})]
# # vector_store.add(documents)

# # # print("Done adding documents")
# # # Query the vector store
# # query = "This is a test document"
# # k = 5
# # results = vector_store.query_with_score(query=query, k=k)

# # print("Results: ", results)

# # results = vector_store.query_directly(query=query, k=k, direct="ex1")

# # print("Results: ", results)

# docs = WebLoader("https://viajsc.com")

# documents = docs.loads()

# documents = SemanticSplitter(embedding=embedding_model).splits(documents=documents)

# print(documents)

# vector_store.add(documents)

# # # # Create a retriever object
# # retriever = vector_store.create_retriever()

# # # # Query the retriever object
# # query = "location of VIA"
# # k = 5
# # results = retriever.invoke(query)

# # print("Results: ", results)

# query = "What is location of VIA"
# k = 5
# results = vector_store.query_with_score(query=query, k=k)

# print("Results: ", results)

from llm import GroqLLM, GeminiLLM
from query_router.router import QueryRouter

llm = GroqLLM(model_name="Gemma2-9b-It", api_key="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").llm

queryRouter = QueryRouter(llm=llm, topic="Sổ tay sinh viên Thủy Lợi").question_router

print(queryRouter.invoke("Trường Đại học Thủy Lợi ở đâu?"))