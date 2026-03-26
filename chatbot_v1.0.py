import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


load_dotenv()
history_file = os.getenv("HISTORY_FILE")

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

  # 1. Khởi tạo Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model=embedded_model,
    google_api_key=embedded_key, 
    trunk_size=trunk_size,
    trunk_overlap=trunk_overlap
    )
    
    # 2. NẠP VECTOR STORE TỪ Ổ CỨNG
vectorstore = FAISS.load_local(
        "faiss_index_db", 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    #Cấu hình Retriever & Chain
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.2}
    )

def save_history_to_local(history):
    # Chuyển đổi list Object thành list Dictionary
    data = []
    for msg in history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        data.append({"role": role, "content": msg.content})
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_history_from_local():
    if not os.path.exists(history_file):
        return []
    with open(history_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    history = []
    for item in data:
        if item["role"] == "user":
            history.append(HumanMessage(content=item["content"]))
        else:
            history.append(AIMessage(content=item["content"]))
    return history

def doc_format(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def summarize_chat_history(chat_history):
    if not chat_history:
        return ""
    # Tạo nội dung để tóm tắt
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
    prompt = f"Hãy tóm tắt nội dung chính của cuộc hội thoại sau đây một cách ngắn gọn để làm ngữ cảnh cho câu hỏi tiếp theo:\n\n{history_str}"
    summary = llm.invoke(prompt)
    return summary.content

def format_chat_history(history):
    formatted_history = ""
    for msg in history:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        formatted_history += f"{role}: {msg.content}\n"
    return formatted_history

#Hàm chatbot không có trí nhớ
def run_chatbot():
    template = (
        "Bạn là một trợ lý thông minh giúp trả lời các câu hỏi dựa trên các tài liệu đã được cung cấp.\n" 
        "Rules:\n"
        "1. Chỉ trả lời dựa trên thông tin có trong tài liệu đã được cung cấp.\n"
        "2. Nếu không tìm thấy thông tin trong tài liệu, hãy trả lời 'Xin lỗi, tôi không có thông tin về câu hỏi này.'\n"
        "3. Trả lời ngắn gọn và chính xác.\n"
        "4. Không đưa ra bất kỳ giả định nào nếu thông tin không có trong tài liệu.\n"  
        "5. Luôn trích tham chiếu nếu có thể (Source, page) sử dụng metadata \n"
        "Context: \n{context}\n\n"
        "Question: {question}\n"
    ) 

    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context":retriever, "question":RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    # 4. Vòng lặp Chat
    while True:
        question = input("Enter your question (go exit de thoat): ")
        if question.lower() == "exit":
            break
        answer = rag_chain.invoke(question)
        print("Answer:", answer)

# Hàm chatbot có trí nhớ (load tất cả history lên)
def run_chatbot_with_memory():
    
     # 4. Vòng lặp Chat
    chat_history = []
    
    print("--- Chatbot voi memory sẵn sàng(Gõ 'exit' để thoát) ---")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == "exit":break

        # 1. Cập nhật Summary nếu lịch sử quá dài
        if len(chat_history) > 6:
            print("--- Đang tối ưu hóa bộ nhớ (Summarizing)... ---")
            chat_history = chat_history[-2:] # Chỉ giữ lại 2 câu gần nhất để duy trì độ mượt

        # 2. Xây dựng Prompt tổng hợp
        # Kết hợp: History cũ + Tài liệu mới tìm được + Câu hỏi hiện tại
        relevant_docs = retriever.invoke(user_input)
        context_text = "\n\n".join([d.page_content for d in relevant_docs])
        
        full_system_prompt = f"""
        Bạn là một trợ lý thông minh giúp trả lời các câu hỏi dựa trên các tài liệu đã được cung cấp. 
        Rules:
        1. Chỉ trả lời dựa trên thông tin có trong tài liệu đã được cung cấp.
        2. Nếu không tìm thấy thông tin trong tài liệu, hãy trả lời 'Xin lỗi, tôi không có thông tin về câu hỏi này.
        3. Trả lời ngắn gọn và chính xác.
        4. Không đưa ra bất kỳ giả định nào nếu thông tin không có trong tài liệu.  
        5. Luôn trích tham chiếu nếu có thể (Source, page) sử dụng metadata.
        Ngữ cảnh tài liệu: {context_text}"""
    
        # 3. Gọi model chính để trả lời
        response = llm.invoke([
            SystemMessage(content=full_system_prompt),
            *chat_history,
            HumanMessage(content=user_input)
        ])

        print(f"\nAI: {response.content}")

        # 4. Lưu vào lịch sử
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response.content))

def run_chatbot_with_summary():
  
     
    # 4. Vòng lặp Chat
    chat_history = load_history_from_local()
    current_summary = ""

    print("--- Chatbot voi Trình tóm tắt (Summary) sẵn sàng(Gõ 'exit' để thoát) ---")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == "exit":break

        # 1. Cập nhật Summary nếu lịch sử quá dài
        if len(chat_history) > 6:
            print("--- Đang tối ưu hóa bộ nhớ (Summarizing)... ---")
            current_summary = summarize_chat_history(chat_history)
            # Sau khi tóm tắt, ta có thể xóa bớt lịch sử cũ để giải phóng RAM/Token
            chat_history = chat_history[-2:] # Chỉ giữ lại 2 câu gần nhất để duy trì độ mượt

        # 2. Xây dựng Prompt tổng hợp
        # Kết hợp: Summary cũ + Tài liệu mới tìm được + Câu hỏi hiện tại
        relevant_docs = retriever.invoke(user_input)
        context_text = "\n\n".join([d.page_content for d in relevant_docs])
        
        full_system_prompt = f"""Bạn là một trợ lý thông minh giúp trả lời các câu hỏi dựa trên các tài liệu đã được cung cấp.
        Rules:
        1. Chỉ trả lời dựa trên thông tin có trong tài liệu đã được cung cấp.
        2. Nếu không tìm thấy thông tin trong tài liệu, hãy trả lời 'Xin lỗi, tôi không có thông tin về câu hỏi này.
        3. Trả lời ngắn gọn và chính xác.
        4. Không đưa ra bất kỳ giả định nào nếu thông tin không có trong tài liệu.  
        5. Luôn trích tham chiếu nếu có thể (Source, page) sử dụng metadata
        Tóm tắt hội thoại trước đó: {current_summary}
        Ngữ cảnh tài liệu: {context_text}"""
    
        # 3. Gọi model chính để trả lời
        response = llm.invoke([
            SystemMessage(content=full_system_prompt),
            *chat_history,
            HumanMessage(content=user_input)
        ])

        print(f"\nAI: {response.content}")

        # 4. Lưu vào lịch sử
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response.content))

        # Giới hạn lịch sử (ví dụ 10 câu cuối) để tránh file quá nặng
        if len(chat_history) > 20: 
            chat_history = chat_history[-20:]
            
        save_history_to_local(chat_history)

if __name__ == "__main__":
    run_chatbot_with_summary()    




