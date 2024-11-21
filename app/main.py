from document_loader import WebLoader
from document_splitter import SemanticSplitter
from embeddings import HFEmbedding
from graph import GraphState, retrieve, wiki_search, route_question
from graph.tool import wiki
from llm import GroqLLM, GeminiLLM
from query_router import QueryRouter
from vector_store import ChromaVectorStore
import streamlit as st
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

### ---


st.sidebar.number_input(
    "Chunk Size",
    min_value=10,
    max_value=1000,
    value=200,
    step=10,
    help="Kích thước của chunk"
)

st.sidebar.number_input(
    "Chunk Overlap",
    min_value=0,
    max_value=1000,
    step=10,
    value=0,
    help="Kích thước của chunk"
)

st.sidebar.number_input(
    "Số lượng tài liệu được truy xuất",
    min_value=1,
    max_value=50,
    value=3,
    step=1,
    help="Số lượng tài liệu sẽ được RAG truy xuất để tạo prompt"
)

header_i = 1
header_text = "{}. Cài đặt ngôn ngữ.".format(header_i)
st.header(header_text)
language_choice = st.radio(
    "Chọn ngôn ngữ",
    ["Tiếng Anh", "Tiếng Việt"],
    index=0
)

header_i += 1
st.header(f"{header_i}. Nguồn tài liệu")
st.subheader(f"{header_i}.1. Tải lên tài liệu", divider=True)
uploaded_files = st.file_uploader(
    "Tải lên tài liệu CSV, JSON, PDF or URL Web",
    type=["csv", "json", "pdf", "txt"],
    accept_multiple_files=True,
    help="Tải lên tài liệu để tạo knowledge base"
)
