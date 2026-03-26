import streamlit as st
import requests

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="R&D Research Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 R&D Intelligence Chatbot")
st.markdown("Hệ thống hỗ trợ nghiên cứu tài liệu dựa trên RAG & Gemini")

# --- KHỞI TẠO STATE (Để lưu lịch sử chat trên trình duyệt) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_summary" not in st.session_state:
    st.session_state.current_summary = ""

# --- SIDEBAR: Cấu hình và Thông tin ---
with st.sidebar:
    st.header("Cấu hình API")
    api_url = st.text_input("API Endpoint", value="http://localhost:8000/chat")
    if st.button("Xóa lịch sử Chat"):
        st.session_state.messages = []
        st.session_state.current_summary = ""
        st.rerun()
    
    st.divider()
    st.markdown("### Tóm tắt ngữ cảnh hiện tại:")
    st.info(st.session_state.current_summary if st.session_state.current_summary else "Chưa có tóm tắt.")

# --- HIỂN THỊ LỊCH SỬ CHAT ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- XỬ LÝ NHẬP LIỆU ---
if prompt := st.chat_input("Nhập câu hỏi nghiên cứu của bạn..."):
    # 1. Hiển thị câu hỏi của User
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Gọi API FastAPI
    with st.chat_message("assistant"):
        with st.spinner("Đang lục tìm tài liệu và suy nghĩ..."):
            try:
                # Đóng gói dữ liệu đúng chuẩn ChatRequest mà bạn đã định nghĩa ở api.py
                payload = {
                    "question": prompt,
                    "history": [
                        {"role": m["role"], "content": m["content"]} 
                        for m in st.session_state.messages[:-1]
                    ],
                    "current_summary": st.session_state.current_summary
                }
                
                response = requests.post(api_url, json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()

                # 3. Lấy kết quả từ API
                answer = data["answer"]
                st.session_state.current_summary = data["new_summary"]
                sources = data["sources"]

                # 4. Hiển thị câu trả lời
                st.markdown(answer)
                
                # Hiển thị nguồn trích dẫn bằng các tag nhỏ
                if sources:
                    st.markdown("---")
                    st.markdown("**Nguồn tham khảo:**")
                    cols = st.columns(len(sources))
                    for i, source in enumerate(sources):
                        cols[i].caption(f"📄 {source}")

                # Lưu vào session state
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except (requests.RequestException, KeyError, ValueError) as e:
                st.error(f"Lỗi kết nối API: {e}")