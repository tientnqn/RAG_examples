import os
import sys
import json
import pytest
from pathlib import Path
from dotenv import load_dotenv
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env từ project root
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# Import các module cần test
from chat_utils import generate_summary, process_chat
from chatbot_v1_0 import save_history_to_local,load_history_from_local,doc_format, format_chat_history, summarize_chat_history

try:
    from api import ChatMessage, ChatRequest, ChatResponse, app
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

# Constants
TEST_HISTORY_FILE = "test_chat_history.json"
PROJECT_HISTORY_FILE = os.getenv("HISTORY_FILE", "chat_history.json")


# ============ FIXTURES ============

@pytest.fixture
def sample_messages():
    """Tạo sample HumanMessage/AIMessage objects."""
    from langchain_core.messages import HumanMessage, AIMessage
    return [
        HumanMessage(content="Xin chào"),
        AIMessage(content="Chào bạn!"),
        HumanMessage(content="Bạn là ai?"),
        AIMessage(content="Tôi là trợ lý AI.")
    ]


@pytest.fixture
def sample_dict_history():
    """Tạo sample history dưới dạng dicts."""
    return [
        {"role": "user", "content": "Xin chào"},
        {"role": "assistant", "content": "Chào bạn!"},
        {"role": "user", "content": "Bạn là ai?"},
        {"role": "assistant", "content": "Tôi là trợ lý AI."}
    ]


@pytest.fixture
def mock_llm():
    """Mock LLM cho generate_summary."""
    llm = Mock()
    llm.invoke = Mock(return_value=Mock(content="Đây là bản tóm tắt hội thoại."))
    return llm


@pytest.fixture
def mock_retriever():
    """Mock retriever cho process_chat."""
    retriever = Mock()
    doc = Mock()
    doc.page_content = "Nội dung tài liệu mẫu"
    doc.metadata = {"source": "test.pdf"}
    retriever.invoke = Mock(return_value=[doc])
    return retriever


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Dọn dẹp file test sau mỗi test."""
    yield
    if os.path.exists(TEST_HISTORY_FILE):
        os.remove(TEST_HISTORY_FILE)


# ============ TEST chat_utils.py ============

class TestGenerateSummary:
    """Test hàm generate_summary."""

    def test_return_old_summary_if_short_history(self, mock_llm):
        """Trả về old_summary nếu history < 4."""
        from langchain_core.messages import HumanMessage
        short_history = [HumanMessage(content="Hi")]
        result = generate_summary(short_history, "old summary", llm=mock_llm)
        assert result == "old summary"
        mock_llm.invoke.assert_not_called()

    def test_generate_summary_with_message_objects(self, mock_llm, sample_messages):
        """Tóm tắt với HumanMessage/AIMessage objects."""
        result = generate_summary(sample_messages, "", llm=mock_llm)
        assert result == "Đây là bản tóm tắt hội thoại."
        # Kiểm tra prompt được gọi đúng
        call_args = mock_llm.invoke.call_args[0][0]
        assert "user: Xin chào" in call_args
        assert "assistant: Chào bạn!" in call_args

    def test_generate_summary_with_dicts(self, mock_llm, sample_dict_history):
        """Tóm tắt với dict history."""
        result = generate_summary(sample_dict_history, "", llm=mock_llm)
        assert result == "Đây là bản tóm tắt hội thoại."
        call_args = mock_llm.invoke.call_args[0][0]
        assert "user: Xin chào" in call_args

    def test_raise_error_if_no_llm(self, sample_messages):
        """Lỗi nếu không truyền llm."""
        with pytest.raises(ValueError, match="llm is required"):
            generate_summary(sample_messages, llm=None)


class TestProcessChat:
    """Test hàm process_chat."""

    def test_process_chat_basic(self, mock_llm, mock_retriever, sample_messages):
        """Test xử lý chat cơ bản."""
        answer, new_summary, sources = process_chat(
            question="Tôi cần thông tin về X?",
            history=sample_messages,
            current_summary="",
            retriever=mock_retriever,
            llm=mock_llm,
            system_prompt_template="Tóm tắt: {updated_summary}\nContext: {context_text}"
        )

        assert answer == "Đây là bản tóm tắt hội thoại."  # from mock_llm
        assert new_summary == ""  # chưa đủ history để tóm tắt
        assert sources == ["test.pdf"]

    def test_triggers_summary_when_history_long(self, mock_llm, mock_retriever, sample_messages):
        # """Kích hoạt tóm tắt khi history >= max_history."""
    
        # # Thiết lập mock_llm trả về: 
        # # Lần 1 (trả lời câu hỏi): "Câu trả lời"
        # # Lần 2 (tóm tắt): "Tóm tắt mới"
        # mock_llm.invoke.side_effect = [
        #     Mock(content="Câu trả lời"), 
        #     Mock(content="Tóm tắt mới")
        # ]

        # long_history = sample_messages * 3

        # answer, new_summary, sources = process_chat(
        #     question="Câu hỏi mới",
        #     history=long_history,
        #     current_summary="",
        #     retriever=mock_retriever,
        #     llm=mock_llm,  # Chỉ dùng 1 LLM duy nhất
        #     system_prompt_template="Context: {context_text}",
        #     max_history=6,
        #     context_limit=2
        # )

        # # Bây giờ assert sẽ PASS vì lần gọi thứ 2 đã trả về đúng giá trị mong muốn
        # assert new_summary == "Tóm tắt mới"
        """Kích hoạt tóm tắt khi history >= max_history sử dụng patch."""
    
        # 1. Setup Mock cho LLM (chỉ dùng để trả lời câu hỏi)
        mock_llm.invoke.return_value = Mock(content="Câu trả lời mẫu")

        # 2. Tạo history dài (4 messages * 3 = 12 messages > max_history=6)
        long_history = sample_messages * 3

        # 3. Patch hàm generate_summary trong module chat_utils
        # Lưu ý: 'chat_utils.generate_summary' là đường dẫn đến hàm bạn muốn mock
        with patch('chat_utils.generate_summary') as mock_gen_summary:
            mock_gen_summary.return_value = "Tóm tắt mới"

            # 4. Gọi hàm với đầy đủ tham số
            answer, new_summary, sources = process_chat(
                question="Câu hỏi mới",
                history=long_history,
                current_summary="",
                retriever=mock_retriever,
                llm=mock_llm,
                system_prompt_template="Context: {context_text}", # Đã thêm đầy đủ
                max_history=6,
                context_limit=2
            )

            # 5. Assertions
            assert answer == "Câu trả lời mẫu"
            assert new_summary == "Tóm tắt mới"
            
            # Kiểm tra xem logic tóm tắt có thực sự được gọi khi history dài không
            mock_gen_summary.assert_called_once()

    def test_history_cleaned_after_summary(self, mock_llm, mock_retriever):
        """History được cắt bớt sau khi tóm tắt."""
        from langchain_core.messages import HumanMessage, AIMessage

        # Tạo history dài
        long_history = [
            HumanMessage(content=f"Message {i}") if i % 2 == 0 else AIMessage(content=f"Response {i}")
            for i in range(10)
        ]

        # Mock llm riêng cho summary
        summary_llm = Mock()
        summary_llm.invoke = Mock(return_value=Mock(content="summary"))

        # Patch generate_summary để dùng mock
        with patch('chat_utils.generate_summary') as mock_generate:
            mock_generate.return_value = "summary"
            answer, new_summary, sources = process_chat(
                question="Test",
                history=long_history,
                current_summary="",
                retriever=mock_retriever,
                llm=mock_llm,
                system_prompt_template="Context: {context_text}",
                max_history=6,
                context_limit=2
            )

            # Kiểm tra generate_summary được gọi
            mock_generate.assert_called_once()

    def test_handles_mixed_message_types(self, mock_llm, mock_retriever):
        """Xử lý đúng cả dict và message objects."""
        mixed_history = [
            {"role": "user", "content": "Dict user"},
            Mock(role="assistant", content="Mock assistant"),
            {"role": "user", "content": "Dict user 2"}
        ]

        answer, new_summary, sources = process_chat(
            question="Test",
            history=mixed_history,
            current_summary="",
            retriever=mock_retriever,
            llm=mock_llm,
            system_prompt_template="Context: {context_text}"
        )

        # Không lỗi và trả về kết quả từ mock_llm
        assert answer == "Đây là bản tóm tắt hội thoại."


# ============ TEST chatbot_v1.0.py ============

class TestHistoryPersistence:
    """Test load/save history."""

    def test_save_and_load_history(self, tmp_path):
        """Lưu và đọc lịch sử từ file."""
        test_file = tmp_path / "test_history.json"

        from langchain_core.messages import HumanMessage, AIMessage
        history = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="How are you?"),
            AIMessage(content="I'm fine, thanks!")
        ]

        # Tạm override history_file
        import chatbot_v1_0
        original_file = chatbot_v1_0.history_file
        chatbot_v1_0.history_file = str(test_file)

        try:
            save_history_to_local(history)
            assert test_file.exists()

            loaded = load_history_from_local()
            assert len(loaded) == 4
            assert isinstance(loaded[0], HumanMessage)
            assert loaded[0].content == "Hello"
            assert isinstance(loaded[1], AIMessage)
            assert loaded[1].content == "Hi there!"
        finally:
            chatbot_v1_0.history_file = original_file

    def test_load_empty_when_file_not_exists(self):
        """Trả về [] nếu file không tồn tại."""
        import chatbot_v1_0
        original_file = chatbot_v1_0.history_file
        chatbot_v1_0.history_file = "non_existent_file.json"

        try:
            result = load_history_from_local()
            assert result == []
        finally:
            chatbot_v1_0.history_file = original_file

    def test_save_history_creates_valid_json(self, tmp_path):
        """File lưu là JSON hợp lệ."""
        test_file = tmp_path / "history.json"

        import chatbot_v1_0
        original_file = chatbot_v1_0.history_file
        chatbot_v1_0.history_file = str(test_file)

        try:
            from langchain_core.messages import HumanMessage
            history = [HumanMessage(content="Test")]
            save_history_to_local(history)

            with open(test_file) as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["role"] == "user"
            assert data[0]["content"] == "Test"
        finally:
            chatbot_v1_0.history_file = original_file


class TestDocFormat:
    """Test hàm doc_format."""

    def test_doc_format_with_docs(self):
        """Format list của documents."""
        doc1 = Mock()
        doc1.page_content = "Content 1"
        doc2 = Mock()
        doc2.page_content = "Content 2"

        result = doc_format([doc1, doc2])
        assert result == "Content 1\n\nContent 2"

    def test_doc_format_empty(self):
        """Format với empty list."""
        result = doc_format([])
        assert result == ""


class TestFormatChatHistory:
    """Test hàm format_chat_history."""

    def test_format_chat_history(self):
        """Format history thành string."""
        from langchain_core.messages import HumanMessage, AIMessage
        history = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi!"),
            HumanMessage(content="How are you?")
        ]

        result = format_chat_history(history)
        assert "User: Hello" in result
        assert "Assistant: Hi!" in result
        assert "User: How are you?" in result


class TestSummarizeChatHistory:
    """Test hàm summarize_chat_history (legacy)."""

    def test_summarize_empty_history(self):
        """Tóm tắt empty history."""
        result = summarize_chat_history([])
        assert result == ""

    def test_summarize_with_llm(self):
        """Tóm tắt với LLM mock."""
        from langchain_core.messages import HumanMessage, AIMessage
        history = [
            HumanMessage(content="Question 1"),
            AIMessage(content="Answer 1")
        ]

        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="Summary text"))

        import chatbot_v1_0
        original_llm = chatbot_v1_0.llm
        chatbot_v1_0.llm = mock_llm

        try:
            result = summarize_chat_history(history)
            assert result == "Summary text"
            mock_llm.invoke.assert_called_once()
        finally:
            chatbot_v1_0.llm = original_llm


# ============ TEST api.py ============

@pytest.mark.skipif(not API_AVAILABLE, reason="api.py not available")
class TestApiSchemas:
    """Test Pydantic schemas."""

    def test_chat_message_valid(self):
        """ChatMessage hợp lệ."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_request_defaults(self):
        """ChatRequest có giá trị mặc định."""
        req = ChatRequest(question="Test")
        assert req.history == []
        assert req.current_summary == ""

    def test_chat_response(self):
        """ChatResponse tạo đúng."""
        resp = ChatResponse(
            answer="Answer text",
            new_summary="Summary",
            sources=["doc1.pdf", "doc2.pdf"]
        )
        assert resp.answer == "Answer text"
        assert resp.new_summary == "Summary"
        assert len(resp.sources) == 2


@pytest.mark.skipif(not API_AVAILABLE, reason="api.py not available")
class TestApiEndpoint:
    """Test FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        """FastAPI test client."""
        from fastapi.testclient import TestClient
        return TestClient(app)

    @pytest.fixture
    def mock_dependencies(self):
        """Mock tất cả dependencies của endpoint."""
        with patch('api.retriever') as mock_retriever, \
             patch('api.llm') as mock_llm, \
             patch('api.process_chat') as mock_process:

            # Setup mock retriever
            doc = Mock()
            doc.page_content = "Mock context"
            doc.metadata = {"source": "mock.pdf"}
            mock_retriever.invoke.return_value = [doc]

            # Setup mock process_chat
            mock_process.return_value = ("Mock answer", "Mock summary", ["mock.pdf"])

            yield {
                'retriever': mock_retriever,
                'llm': mock_llm,
                'process_chat': mock_process
            }

    def test_chat_endpoint_success(self, client, mock_dependencies):
        """POST /chat thành công."""
        payload = {
            "question": "What is RAG?",
            "history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"}
            ],
            "current_summary": ""
        }

        response = client.post("/chat", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["answer"] == "Mock answer"
        assert data["new_summary"] == "Mock summary"
        assert "mock.pdf" in data["sources"]

    def test_chat_endpoint_empty_question(self, client, mock_dependencies):
        """Xử lý question rỗng."""
        payload = {
            "question": "",
            "history": []
        }

        response = client.post("/chat", json=payload)
        assert response.status_code == 200  # Vẫn xử lý được

    def test_chat_endpoint_handles_exception(self, client):
        """Xử lý exception từ dependencies."""
        with patch('api.process_chat', side_effect=Exception("Test error")):
            payload = {"question": "Test"}
            response = client.post("/chat", json=payload)
            assert response.status_code == 500
            assert "Test error" in response.json()["detail"]

    def test_chat_endpoint_history_limit(self, client, mock_dependencies):
        """Kiểm tra history bị giới hạn khi tóm tắt."""
        # Tạo history dài
        long_history = [
            {"role": "user", "content": f"Q{i}"}
            for i in range(10)
        ] + [
            {"role": "assistant", "content": f"A{i}"}
            for i in range(10)
        ]

        payload = {
            "question": "New question",
            "history": long_history,
            "current_summary": ""
        }

        response = client.post("/chat", json=payload)
        assert response.status_code == 200

        # Kiểm tra process_chat được gọi với history đã lọc
        mock_dependencies['process_chat'].assert_called_once()


# ============ TEST embedded.py ============

class TestEmbedded:
    """Test hàm ingest_docs từ embedded.py."""

    def test_ingest_docs_structure(self):
        """Kiểm tra cấu trúc hàm ingest_docs."""
        from embedded import ingest_docs
        assert callable(ingest_docs)

    @patch('embedded.DirectoryLoader')
    @patch('embedded.RecursiveCharacterTextSplitter')
    @patch('embedded.FAISS')
    @patch('embedded.GoogleGenerativeAIEmbeddings')
    def test_ingest_docs_flow(self, mock_embeddings, mock_faiss, mock_splitter, mock_loader):
        """Test luồng ingest_docs với mock."""
        from embedded import ingest_docs

        # Mock documents
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        mock_loader.return_value.load.return_value = [mock_doc]

        # Mock text splitter
        mock_split = Mock()
        mock_split.split_documents.return_value = [mock_doc]
        mock_splitter.return_value = mock_split

        # Mock FAISS
        mock_vectorstore = Mock()
        mock_vectorstore.save_local.return_value = None
        mock_faiss.from_documents.return_value = mock_vectorstore

        # Gọi hàm
        ingest_docs()

        # Kiểm tra các bước
        mock_loader.assert_called_once()
        mock_splitter.assert_called_once()
        mock_split.split_documents.assert_called_once_with([mock_doc])
        mock_faiss.from_documents.assert_called_once()
        mock_vectorstore.save_local.assert_called_once_with("faiss_index_db")


# ============ TEST app.py ============

class TestApp:
    """Test app.py (Streamlit UI)."""

    def test_app_file_exists(self):
        """Kiểm tra file app.py tồn tại."""
        from pathlib import Path
        app_path = Path(__file__).parent.parent / "app.py"
        assert app_path.exists()

    def test_payload_structure(self):
        """Test cấu trúc payload khi gọi API."""
        # Mô phỏng messages và summary
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"}
        ]
        current_summary = "Previous summary"

        prompt = "New question"
        payload = {
            "question": prompt,
            "history": [
                {"role": m["role"], "content": m["content"]}
                for m in messages[:-1]
            ],
            "current_summary": current_summary
        }

        assert payload["question"] == "New question"
        assert len(payload["history"]) == 1  # messages[:-1]
        assert payload["history"][0]["role"] == "user"
        assert payload["current_summary"] == "Previous summary"

    def test_app_has_chat_input(self):
        """Kiểm tra app.py có chứa chat_input logic."""
        from pathlib import Path
        app_path = Path(__file__).parent.parent / "app.py"
        content = app_path.read_text(encoding='utf-8')
        assert 'st.chat_input' in content
        assert 'requests.post' in content


# ============ TEST INTEGRATION ============

class TestIntegration:
    """Test tích hợp giữa các module."""

    def test_full_chat_flow(self, tmp_path):
        """Mô phỏng full chat flow."""
        from langchain_core.messages import HumanMessage, AIMessage

        # 1. Setup test history file
        test_history_file = tmp_path / "integration_test.json"

        import chatbot_v1_0
        original_file = chatbot_v1_0.history_file
        chatbot_v1_0.history_file = str(test_history_file)

        try:
            # 2. Tạo history ban đầu
            initial_history = [
                HumanMessage(content="Tiếng Việt có thể được sử dụng?"),
                AIMessage(content="Có, tôi có thể trả lời bằng tiếng Việt.")
            ]
            save_history_to_local(initial_history)

            # 3. Load lại
            loaded = load_history_from_local()
            assert len(loaded) == 2

            # 4. Test với process_chat (sử dụng mock)
            mock_retriever = Mock()
            doc = Mock()
            doc.page_content = "Test content"
            doc.metadata = {"source": "test.pdf"}
            mock_retriever.invoke.return_value = [doc]

            mock_llm = Mock()
            mock_llm.invoke = Mock(return_value=Mock(content="Test answer"))

            answer, summary, sources = process_chat(
                question="Test question",
                history=loaded,
                current_summary="",
                retriever=mock_retriever,
                llm=mock_llm,
                system_prompt_template="Context: {context_text}"
            )

            assert answer == "Test answer"
            assert sources == ["test.pdf"]

        finally:
            chatbot_v1_0.history_file = original_file


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
