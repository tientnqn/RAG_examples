import streamlit as st
import requests
import urllib.parse
import json
import base64

# --- 1. CONFIG & CONSTANTS ---
st.set_page_config(page_title="R&D Research Chatbot", page_icon="🤖", layout="wide")
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REDIRECT_URI = st.secrets.get("GOOGLE_REDIRECT_URI", "http://localhost:8501")

# --- 2. AUTH UTILS ---
def get_auth_url() -> str:
    params = {
        "client_id": st.secrets["GOOGLE_CLIENT_ID"],
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "online", "prompt": "consent",
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"

def get_user_info(code: str) -> dict:
    data = {
        "code": code,
        "client_id": st.secrets["GOOGLE_CLIENT_ID"],
        "client_secret": st.secrets["GOOGLE_CLIENT_SECRET"],
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    try:
        resp = requests.post(TOKEN_URL, data=data).json()
        payload = resp["id_token"].split(".")[1]
        return json.loads(base64.urlsafe_b64decode(payload + "==").decode())
    except: return {}

# --- 3. AUTH LOGIC ---
if "code" in st.query_params:
    user = get_user_info(st.query_params.pop("code"))
    if user: st.session_state.user = user
    st.rerun()

if "user" not in st.session_state:
    st.title("🤖 R&D Intelligence Chatbot")
    st.warning("Vui lòng đăng nhập để sử dụng hệ thống RAG nội bộ.")
    st.markdown(f'''<a href="{get_auth_url()}" target="_parent" style="display:inline-block;padding:12px 24px;background:#4285F4;color:white;text-decoration:none;border-radius:8px;font-weight:bold;">🔐 Login with Google</a>''', unsafe_allow_html=True)
    st.stop()

# --- 4. SESSION STATE INIT ---
for key, default in {"messages": [], "summary": ""}.items():
    st.session_state.setdefault(key, default)

# --- 5. SIDEBAR UI (PHẦN SẾP CẦN THAY THẾ) ---
with st.sidebar:
    # Avatar tròn bằng CSS
    user = st.session_state.user
    st.markdown("""
        <style>
        .profile-pic { width: 60px; height: 60px; border-radius: 50%; object-fit: cover; border: 2px solid #4285F4; }
        </style>
    """, unsafe_allow_html=True)
    
    col_av, col_txt = st.columns([1, 3])
    with col_av:
        st.markdown(f'<img src="{user.get("picture")}" class="profile-pic">', unsafe_allow_html=True)
    with col_txt:
        st.markdown(f"**{user.get('name')}**")
        st.caption(user.get("email"))

    st.divider()
    api_url = st.text_input("🚀 API Endpoint", "http://localhost:8000/chat")
    
    c1, c2 = st.columns(2)
    if c1.button("🗑️ Clear", use_container_width=True):
        st.session_state.messages, st.session_state.summary = [], ""
        st.rerun()
    if c2.button("🚪 Logout", use_container_width=True):
        del st.session_state.user
        st.rerun()

    st.divider()
    st.subheader("📝 Context Summary")
    st.info(st.session_state.summary or "Chưa có tóm tắt.")

# --- 6. MAIN CHAT UI ---
st.title("🤖 R&D Intelligence Chatbot")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Nhập câu hỏi nghiên cứu..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {"question": prompt, "history": st.session_state.messages[:-1], "current_summary": st.session_state.summary}
                res = requests.post(api_url, json=payload, timeout=30).json()
                
                st.session_state.summary = res["new_summary"]
                st.markdown(res["answer"])
                
                if res.get("sources"):
                    with st.expander("📚 Nguồn tham khảo"):
                        st.write(", ".join([f"📄 {s}" for s in res["sources"]]))
                
                st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
            except Exception as e:
                st.error(f"API Error: {str(e)}")