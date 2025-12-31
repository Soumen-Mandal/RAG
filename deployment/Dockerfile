FROM python:3.12-slim

# 1️⃣ Install system dependencies (CRITICAL)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 2️⃣ Set workdir
WORKDIR /app

# 3️⃣ Copy requirements
COPY requirements.txt .

# 4️⃣ Upgrade pip tools
RUN pip install --upgrade pip setuptools wheel

# 5️⃣ Install Python dependencies (slow but correct)
RUN pip install --no-cache-dir -r requirements.txt

# 6️⃣ Copy application code
COPY . .

# 7️⃣ Cloud Run config
ENV PORT=8080
EXPOSE 8080

# 8️⃣ Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
