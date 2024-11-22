from document_loader import WebLoader, CSVLoader, JSLoader, PDFLoader  
from document_splitter import SemanticSplitter, RSTextSplitter
from embeddings import HFEmbedding
from graph.graphState import GraphState
from graph.tool import wiki
from llm import GroqLLM, GeminiLLM
from query_router import QueryRouter
from vector_store import ChromaVectorStore
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def clear_session_state():
    for key in st.session_state.keys():
        del st.session_state[key]


### 
st.set_page_config(page_title="Chatbot", page_icon=":dog:", layout="centered")
st.title("Chatbot của Đinh Văn Linh")
st.caption("Đinh Văn Linh - 2051063562 - K62 - Đại học Thủy Lợi")
st.markdown(""" <style>
.stSidebar {
            background-color: #999999;
            color: #333}""", unsafe_allow_html=True)
st.sidebar.image("./via-logo.png", use_container_width=True)
st.sidebar.title("Cấu hình chatbot")
st.sidebar.caption("Điều chỉnh ít thôi không lag đó !!!")

### --- Khởi tạo session --- ###

if "language" not in st.session_state:
    st.session_state.language = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 200
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 10

### --- Sidebar --- ###


st.session_state.chunk_size = st.sidebar.number_input(
    "Chunk Size",
    min_value=10,
    max_value=1000,
    value=200,
    step=10,
    help="Kích thước của chunk"
)

st.session_state.chunk_overlap = st.sidebar.number_input(
    "Chunk Overlap",
    min_value=0,
    max_value=1000,
    step=10,
    value=0,
    help="Kích thước của chunk"
)

st.session_state.number_docs_retrieval = st.sidebar.number_input(
    "Số lượng tài liệu được truy xuất",
    min_value=1,
    max_value=50,
    value=3,
    step=1,
    help="Số lượng tài liệu sẽ được RAG truy xuất để tạo prompt"
)


### --- Cài đặt ngôn ngữ --- ###

header_i = 1
header_text = "{}. Cài đặt ngôn ngữ.".format(header_i)
st.header(header_text)
language_choice = st.radio(
    "Chọn ngôn ngữ",
    ["Tiếng Anh", "Tiếng Việt"],
    index=0
)

if language_choice == "Tiếng Anh":
    if st.session_state.language != "en":
        st.session_state.language = "en"
        if st.session_state.get("embedding_model_name") != "all-mpnet-base-v2":
            st.session_state.embedding_model = HFEmbedding("all-mpnet-base-v2")
            st.session_state.embedding_model_name = "all-mpnet-base-v2"
        st.success("Đã chọn ngôn ngữ Tiếng Anh với model: all-mpnet-base-v2")

elif language_choice == "Tiếng Việt":
    if st.session_state.language != "vi":
        st.session_state.language = "vi"
        if st.session_state.get("embedding_model_name") != "keepitreal/vietnamese-sbert":
            st.session_state.embedding_model = HFEmbedding("keepitreal/vietnamese-sbert")
            st.session_state.embedding_model_name = "keepitreal/vietnamese-sbert"
        st.success("Đã chọn ngôn ngữ Tiếng Việt với model: keepitreal/vietnamese-sbert")


### --- Nguồn tài liệu --- ###

header_i += 1
st.header(f"{header_i}. Nguồn tài liệu")
st.subheader(f"{header_i}.1. Tải lên tài liệu", divider=True)
uploaded_files = st.file_uploader(
    "Tải lên tài liệu CSV, JSON, PDF or URL Web",
    type=["csv", "json", "pdf", "txt"],
    accept_multiple_files=True,
    help="Tải lên tài liệu để tạo knowledge base"
)

st.session_state.data_saved_success = False

if uploaded_files is not None:
    all_data = []

    for uploaded_file in uploaded_files:
        #lưu uploaded_file vào data/document và lấy đc đường dẫn
        with open(f"data/documents/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Đã lưu tài liệu {uploaded_file.name} vào data/documents")
        
        file_path = f"data/documents/{uploaded_file.name}"

        if uploaded_file.name.endswith(".csv"):
            docs = CSVLoader(file_path).loads()
            all_data.append(docs)
        elif uploaded_file.name.endswith(".json"):
            docs = JSLoader(file_path).loads()
            all_data.append(docs)
        elif uploaded_file.name.endswith(".pdf"):
            docs = PDFLoader(file_path).loads()
            all_data.append(docs)
        else:
            st.error("File không hợp lệ")

    if all_data:

        chunk_option = st.selectbox(
            "Chọn cách chia document",
            ["None" ,"Recursive Splitter", "Semantic Splitter"]
        )
        print(chunk_option)
            
        if chunk_option == "Recursive Splitter":
            st.session_state.chunk_option = "Recursive Splitter"
            documents = RSTextSplitter(documents=all_data[0]).split_documents(
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap
            )
            st.success("Đã chia document theo rsTextSplitter")
                # Hiển thị document sau khi split
            # st.text_area("Tài liệu sau khi chunk",documents, height=200)
            
        
        elif chunk_option == "Semantic Splitter":
            st.session_state.chunk_option = "Semantic Splitter"


            embeddings = st.session_state.embedding_model
            documents = SemanticSplitter(embedding=embeddings).splits(
                all_data[0]
            )
            st.success("Đã chia document theo semanticSplitter")
                # Hiển thị document sau khi split
            # st.text_area("Tài liệu sau khi chunk",documents, height=200)
        chunk_records = []

        for doc in documents:
            chunk_records.append(doc.page_content)
        # print(chunk_records)
        chunk_records = pd.DataFrame(chunk_records)
        # print("pass")
        st.write("Tài liệu sau khi chunk với ", len(chunk_records), "chunks")

        st.dataframe(chunk_records, use_container_width=True, height=300)
        
    if st.button("Lưu tài liệu vào vector store"):
        vector_store = ChromaVectorStore("documents", st.session_state.embedding_model)
        vector_store.add(documents)
        st.success("Đã lưu tài liệu vào vector store")
        st.session_state.data_saved_success = True
