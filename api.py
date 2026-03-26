import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

# Cấu hình từ .env
embedded_key = os.getenv("GOOGLE_EMBEDDING_API_KEY")
trunk_size = os.getenv("TRUNK_SIZE")
trunk_overlap = os.getenv("TRUNK_OVERLAP")
embedded_model = os.getenv("GOOGLE_EMBEDDING_MODEL")
chat_model = os.getenv("GEMINI_MODEL")
chat_api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    model=chat_model,
    google_api_key=chat_api_key,
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

# --- 3. HÀM Tom TẮT (HELPER) ---
def generate_summary(history: List[ChatMessage], old_summary: str = ""):
    if len(history) < 4: # Chỉ tóm tắt nếu lịch sử bắt đầu dài (vượt quá 2 cặp câu hỏi-đáp)
        return old_summary
    
    history_text = "\n".join([f"{m.role}: {m.content}" for m in history])
    prompt = f"""Dựa trên bản tóm tắt cũ: "{old_summary}" 
    và các tin nhắn mới sau đây:
    {history_text}
    Hãy tóm tắt nội dung chính của cuộc hội thoại sau đây một cách ngắn gọn để làm ngữ cảnh cho câu hỏi tiếp theo."""
    
    response = llm.invoke(prompt)
    return response.content

# --- 4. ENDPOINT CHÍNH ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Bước A: Cập nhật Summary nếu lịch sử dài
        updated_summary = request.current_summary
        if len(request.history) >= 6:
            updated_summary = generate_summary(request.history, request.current_summary)

        # Bước B: Tìm kiếm tài liệu (RAG)
        docs = retriever.invoke(request.question)
        context_text = "\n\n".join([d.page_content for d in docs])
        sources = list(set([d.metadata.get("source", "Unknown") for d in docs]))

        # Bước C: Xây dựng Prompt tổng hợp
        # system_instruction = (
        #     "Bạn là trợ lý R&D chuyên nghiệp.\n"
        #     f"Tóm tắt các nội dung đã thảo luận trước đó: {updated_summary}\n"
        #     f"Dưới đây là ngữ cảnh từ tài liệu PDF: {context_text}\n"
        #     "Hãy trả lời câu hỏi dựa trên thông tin trên. Nếu không có, hãy báo không biết."
        # )

        # Sửa lại cách nối chuỗi trong Python để tránh dư thừa dấu ngoặc kép
        system_instruction = f"""Bạn là một trợ lý thông minh.
        Nhiệm vụ: Trả lời câu hỏi dựa trên Ngữ cảnh tài liệu được cung cấp.
        QUY TẮC:
        1. Chỉ sử dụng thông tin trong Ngữ cảnh tài liệu.
        2. Trả lời ngắn gọn, tập trung vào từ khóa chính.
        3. Nếu tài liệu mô tả đặc điểm của đối tượng, hãy dùng các đặc điểm đó để trả lời câu hỏi định nghĩa.
        4. Trích dẫn nguồn nếu có.

        Tóm tắt hội thoại: {updated_summary}
        Ngữ cảnh tài liệu: {context_text}"""

        # Bước D: Gọi model trả lời (Chỉ gửi 2 câu gần nhất + System Prompt để tối ưu)
        messages = [SystemMessage(content=system_instruction)]
        # Chỉ lấy tối đa 2 tin nhắn gần nhất từ history để duy trì ngữ cảnh tức thời
        for msg in request.history[-2:]:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            else:
                messages.append(AIMessage(content=msg.content))
        
        messages.append(HumanMessage(content=request.question))

        # --- THÊM DÒNG NÀY ĐỂ SOI DỮ LIỆU ---
        print("="*50)
        print(f"DEBUG CONTEXT FOR QUESTION: {request.question}")
        print(messages)
        print("="*50)
        ai_response = llm.invoke(messages)

        return ChatResponse(
            answer=ai_response.content,
            new_summary=updated_summary,
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)