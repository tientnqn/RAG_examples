# Common chat processing logic for RAG-based chatbot
from typing import List, Tuple
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


def generate_summary(history, old_summary: str = "", llm=None) -> str:
    """
    Tạo tóm tắt cuộc hội thoại.

    Args:
        history: Danh sách tin nhắn (HumanMessage, AIMessage, hoặc dict)
        old_summary: Tóm tắt cũ để nối tiếp
        llm: Đối tượng LLM dùng để tóm tắt

    Returns:
        Tóm tắt mới
    """
    if len(history) < 4:  # Chỉ tóm tắt nếu lịch sử đủ dài
        return old_summary

    lines = []
    for m in history:
        if isinstance(m, HumanMessage):
            role = "user"
            content = m.content
        elif isinstance(m, AIMessage):
            role = "assistant"
            content = m.content
        elif isinstance(m, dict):
            role = m.get('role', 'unknown')
            content = m.get('content', '')
        else:
            role = "unknown"
            content = str(m)
        lines.append(f"{role}: {content}")
    history_text = "\n".join(lines)

    prompt = f"""Dựa trên bản tóm tắt cũ: "{old_summary}"
và các tin nhắn mới sau đây:
{history_text}
Hãy tóm tắt nội dung chính của cuộc hội thoại sau đây một cách ngắn gọn để làm ngữ cảnh cho câu hỏi tiếp theo."""

    if llm is None:
        raise ValueError("llm is required to generate summary")

    response = llm.invoke(prompt)
    return response.content


def process_chat(
    question: str,
    history: List,
    current_summary: str,
    retriever,
    llm,
    system_prompt_template: str,
    max_history: int = 6,
    context_limit: int = 2
) -> Tuple[str, str, List[str]]:
    """
    Xử lý logic chung cho chatbot.

    Args:
        question: Câu hỏi của người dùng
        history: Lịch sử chat (list of messages/dicts)
        current_summary: Tóm tắt hiện tại
        retriever: Đối tượng retriever
        llm: Đối tượng LLM
        system_prompt_template: Mẫu prompt hệ thống (có thể dùng .format)
        max_history: Ngưỡng số tin nhắn để kích hoạt tóm tắt
        context_limit: Số tin nhắn gần nhất giữ lại khi tóm tắt

    Returns:
        Tuple (answer, new_summary, sources)
    """
    # Bước A: Cập nhật summary nếu lịch sử dài
    updated_summary = current_summary
    if len(history) >= max_history:
        updated_summary = generate_summary(history, current_summary, llm=llm)
        # Lọc lại lịch sử để chỉ giữ số lượng context_limit cho turn tiếp theo
        history = history[-context_limit:]

    # Bước B: Tìm kiếm tài liệu (RAG)
    docs = retriever.invoke(question)
    context_text = "\n\n".join([d.page_content for d in docs])
    sources = list(set([d.metadata.get("source", "Unknown") for d in docs]))

    # Bước C: Xây dựng prompt hệ thống
    full_system_prompt = system_prompt_template.format(
        updated_summary=updated_summary,
        context_text=context_text
    )

    # Bước D: Tạo messages và gọi LLM
    messages = [SystemMessage(content=full_system_prompt)]

    # Thêm lịch sử hội thoại gần nhất (đảm bảo đúng role)
    for msg in history[-context_limit:]:
        if isinstance(msg, HumanMessage):
            messages.append(HumanMessage(content=msg.content))
        elif isinstance(msg, AIMessage):
            messages.append(AIMessage(content=msg.content))
        elif isinstance(msg, dict):
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=str(msg)))

    messages.append(HumanMessage(content=question))

    ai_response = llm.invoke(messages)

    return ai_response.content, updated_summary, sources
