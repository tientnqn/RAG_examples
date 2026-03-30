# CLAUDE.md - Hướng dẫn cho AI Agent

## 🛠 Lệnh vận hành (Build & Run)
- Cài đặt môi trường: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
- Chạy ứng dụng: `uvicorn main:app --reload`
- Cài đặt package mới: `pip install <package_name> && pip freeze > requirements.txt`

## 🧪 Lệnh kiểm thử (Testing)
- Chạy toàn bộ test: `pytest`
- Chạy một file cụ thể: `pytest tests/test_filename.py`
- Chạy test với coverage: `pytest --cov=app tests/`

## 📏 Quy tắc lập trình (Code Style Guidelines)
- **Ngôn ngữ:** Viết code và comment bằng tiếng Anh. Giải thích cho người dùng bằng tiếng Việt.
- **Kiểu dữ liệu:** Luôn sử dụng Python Type Hints (ví dụ: `def func(name: str) -> bool:`)
- **Naming:** - Biến và hàm: `snake_case`
  - Class: `PascalCase`
  - Constant: `UPPER_SNAKE_CASE`
- **Cấu trúc:** Ưu tiên sử dụng Async/Await cho các tác vụ I/O (Database, API call).
- **Xử lý lỗi:** Luôn bao bọc các hàm quan trọng bằng `try-except` và log lỗi rõ ràng.
- **Imports:** Nhóm imports theo thứ tự: Thư viện chuẩn, Thư viện bên thứ ba, Modules của project.
- **Linting:** Tuân thủ chuẩn PEP 8.
