# 1. Sử dụng Python 3.12 bản nhẹ
FROM python:3.12-slim

# 2. Thiết lập môi trường
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Cài đặt thư viện hệ thống cần thiết cho FAISS và PDF
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Tạo thư mục làm việc
WORKDIR /app

# 5. Cài đặt Poetry
RUN pip install poetry

# 6. Copy các file quản lý thư viện trước
COPY pyproject.toml poetry.lock* ./

# 7. Cài đặt thư viện trực tiếp vào hệ thống của Container
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# 8. Copy toàn bộ code và dữ liệu vào (vì cấu trúc của bạn là Flat)
COPY . .

# 9. Lệnh khởi chạy (Mặc định chạy API, bạn có thể đổi thành app.py nếu muốn)
CMD ["python", "api.py"]