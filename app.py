import os
from dotenv import load_dotenv
import streamlit as st
import requests
import urllib.parse

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="R&D Research Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 R&D Intelligence Chatbot")
st.markdown("Hệ thống hỗ trợ nghiên cứu tài liệu dựa trên RAG & Gemini")


# --- HÀM GOOGLE AUTH ---
def get_google_auth_url():
    """Tạo URL xác thực Google OAuth2."""
    client_id = st.secrets["GOOGLE_CLIENT_ID"]
    auth_url = "https://accounts.google.com/o/oauth2/v2/auth"
    params = {
        "client_id": client_id,
        "redirect_uri": st.secrets.get("GOOGLE_REDIRECT_URI", "http://localhost:8501"),
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "online",
        "prompt": "consent",
    }
    return auth_url + "?" + urllib.parse.urlencode(params)  # type: ignoreencode(params)  # type: ignore


def exchange_code_for_user_info(code: str) -> dict | None:
    """Đổi authorization code để lấy user info."""
    token_url = "https://oauth2.googleapis.com/token"
    token_data = {
        "code": code,
        "client_id": st.secrets["GOOGLE_CLIENT_ID"],
        "client_secret": st.secrets["GOOGLE_CLIENT_SECRET"],
        "redirect_uri": st.secrets.get("GOOGLE_REDIRECT_URI", "http://localhost:8501"),
        "grant_type": "authorization_code",
    }
    token_response = requests.post(token_url, data=token_data)
    token_json = token_response.json()

    if "id_token" not in token_json:
        return None

    # Decode JWT id_token để lấy email
    import base64
    import json

    id_token = token_json["id_token"]
    payload = id_token.split(".")[1]
    # Pad base64
    payload += "=" * (4 - len(payload) % 4)
    user_info = json.loads(base64.urlsafe_b64decode(payload))
    return user_info


# --- KIỂM TRA AUTH ---
# 1. Xử lý callback từ Google OAuth
query_params = st.query_params
if "code" in query_params:
    code = query_params["code"]
    user_info = exchange_code_for_user_info(code)
    if user_info:
        st.session_state["user"] = {
            "email": user_info.get("email"),
            "name": user_info.get("name"),
            "picture": user_info.get("picture"),
        }
    # Xóa query params để reset URL
    st.query_params.clear()
    st.rerun()

# 2. Nếu chưa đăng nhập → hiện login page
if "user" not in st.session_state:
    st.markdown("---")
    st.warning("Vui lòng đăng nhập bằng Google Account để sử dụng chatbot")

    # Tạo URL login bằng hàm
    login_url = get_google_auth_url()

    # Redirect trong cùng tab bằng target="_parent"
    st.markdown(
        f'<a href="{login_url}" target="_parent" style="display: inline-block; padding: 10px 20px; background-color: #4285F4; color: white; text-decoration: none; border-radius: 5px; text-align: center;">🔐 Login with Google</a>',
        unsafe_allow_html=True
    )

    st.stop()

# --- Nếu đã đăng nhập → hiện chatbot ---

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

    # Hiển thị thông tin user
    user = st.session_state.get("user", {})
    st.markdown(f"**User:** {user.get('name', 'User')}")
    st.caption(user.get("email", ""))

    if st.button("Đăng xuất"):
        del st.session_state["user"]
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
                    "current_summary": st.session_state.current_summary,
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
