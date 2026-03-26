import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()
MARKDOWN_SEPARATORS = [
    # 1. Các cấp độ tiêu đề (H1 đến H6)
    r"\n#{1,6} ", 
    
    # 2. Đường kẻ ngang phân cách (Horizontal Rules)
    r"\n---+\n", 
    r"\n\*+\n",
    
    # 3. Các khối code lớn (Code Blocks)
    r"\n```\n", 
    
    # 4. Ngắt đoạn văn bản (Double Newline)
    "\n\n", 
    
    # 5. Danh sách (List items: *, -, +, hoặc 1.)
    r"\n\s*[\*\-\+] ", 
    r"\n\s*\d+\. ", 
    
    # 6. Trích dẫn (Blockquotes)
    r"\n> ",
    
    # 7. Ngắt dòng đơn, dấu chấm, dấu phẩy và khoảng trắng (Cuối cùng)
    "\n", 
    ". ", 
    ", ", 
    " ", 
    ""
]

# Cấu hình từ .env
embedded_key = os.getenv("GOOGLE_EMBEDDING_API_KEY")
trunk_size = os.getenv("TRUNK_SIZE")
trunk_overlap = os.getenv("TRUNK_OVERLAP")
embedded_model = os.getenv("GOOGLE_EMBEDDING_MODEL")

def ingest_docs():
    # 1. Load tài liệu
    loader = DirectoryLoader(
    './papers', 
    glob='**/*.pdf', 
    show_progress=True, 
    silent_errors=True, 
    loader_cls=PyPDFLoader,
    use_multithreading=True
    )
    documents = loader.load()

    # 2. Split văn bản (Dùng separators của bạn)
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200, 
    chunk_overlap=200,
    add_start_index=True,
    strip_whitespace=True,
    separators=MARKDOWN_SEPARATORS)

    texts = text_splitter.split_documents(documents)
    # 3. Khởi tạo Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
    model=embedded_model,
    google_api_key=embedded_key, 
    trunk_size=trunk_size,
    trunk_overlap=trunk_overlap
    )

    # 4. Tạo Vector Store và LƯU XUỐNG Ổ CỨNG
    print("Đang tạo vector database... Vui lòng đợi.")
    vectorstore = FAISS.from_documents(
    documents = texts, 
    embedding=embeddings,
    distance_strategy=DistanceStrategy.COSINE
    )
    vectorstore.save_local("faiss_index_db")
    print("Đã lưu thành công vào thư mục 'faiss_index_db'")

if __name__ == "__main__":
    ingest_docs()








