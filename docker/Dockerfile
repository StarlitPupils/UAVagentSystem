# Dockerfile - UAVagent 2.0
FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 项目文件
COPY . .

# 创建必要目录
RUN mkdir -p models output logs

# 暴露API端口
EXPOSE 8000

# 默认启动API服务器
CMD ["python", "api/server.py"]
