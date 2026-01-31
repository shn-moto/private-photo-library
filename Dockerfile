# PyTorch 2.6 с CUDA 12.4 (для transformers)
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Установить переменные окружения
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Warsaw

# Установить рабочую директорию
WORKDIR /app

# Установить системные зависимости и timezone
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    git \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
    tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# Копировать requirements.txt
COPY requirements.txt .

# Установить Python зависимости (torch уже в образе!)
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Копировать весь код проекта
COPY . .

# Создать директорию для логов и сделать скрипты исполняемыми
RUN mkdir -p logs cache && \
    chmod +x /app/scripts/*.sh 2>/dev/null || true

# Expose порт для API
EXPOSE 8000

# По умолчанию запускаем основной сервис индексирования
CMD ["python", "main.py"]
