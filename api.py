import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chat_utils import process_chat
from langchain_openai import ChatOpenAI

load_dotenv()

# Cấu hình từ .env
embedded_key = os.getenv("GOOGLE_EMBEDDING_API_KEY")
trunk_size = os.getenv("TRUNK_SIZE")
trunk_overlap = os.getenv("TRUNK_OVERLAP")
embedded_model = os.getenv("GOOGLE_EMBEDDING_MODEL")
chat_model = os.getenv("GEMINI_MODEL")
chat_api_key = os.getenv("GEMINI_API_KEY")
base_url = os.getenv("BASE_URL")

llm = ChatGoogleGenerativeAI(
    model=chat_model,
    google_api_key=chat_api_key,
    temperature=0
    )

llm_openai_local = ChatOpenAI(
    base_url=base_url,
    api_key="not needed",
    model="not needed",
    temperature=0
)

embeddings = GoogleGenerativeAIEmbeddings(
    model=embedded_model,
    google_api_key=embedded_key, 
    trunk_size=trunk_size,
    trunk_overlap=trunk_overlap
    )

app = FastAPI(title="RAG Research API v1", description="API tích hợp tự động tóm tắt hội thoại (Conversation Summary)")

#Khoi tao vectorstore tu o cung
vectorstore = FAISS.load_local(
        "faiss_index_db", 
        embeddings, 
        allow_dangerous_deserialization=True
    )


retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.2}
    )

# --- 2. ĐỊNH NGHĨA SCHEMAS ---
class ChatMessage(BaseModel):
    role: str  # "user" hoặc "assistant"
    content: str

class ChatRequest(BaseModel):
    question: str
    history: List[ChatMessage] = []
    current_summary: Optional[str] = "" # Client có thể gửi lại summary cũ để nối tiếp

class ChatResponse(BaseModel):
    answer: str
    new_summary: str # Trả lại summary mới cho client lưu trữ
    sources: List[str]

# --- 4. ENDPOINT CHÍNH ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Sử dụng hàm process_chat từ chat_utils
        answer, new_summary, sources = process_chat(
            question=request.question,
            history=request.history,
            current_summary=request.current_summary,
            retriever=retriever,
            llm=llm,
            system_prompt_template="""Bạn là một trợ lý thông minh.
            Nhiệm vụ: Trả lời câu hỏi dựa trên Ngữ cảnh tài liệu được cung cấp.
            QUY TẮC:
            1. Chỉ sử dụng thông tin trong Ngữ cảnh tài liệu.
            2. Trả lời ngắn gọn, tập trung vào từ khóa chính.
            3. Nếu tài liệu mô tả đặc điểm của đối tượng, hãy dùng các đặc điểm đó để trả lời câu hỏi định nghĩa.
            4. Trích dẫn nguồn nếu có.

            Tóm tắt hội thoại: {updated_summary}
            Ngữ cảnh tài liệu: {context_text}""",
            max_history=6,
            context_limit=2
        )

        return ChatResponse(
            answer=answer,
            new_summary=new_summary,
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)