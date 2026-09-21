# MaintainAI vLLM Inference Server Dockerfile
# Build: docker build -t maintainai-vllm .
# Run: docker run --gpus all -p 8000:8000 -v /path/to/adapter:/adapter maintainai-vllm --lora-path /adapter

FROM nvidia/cuda:12.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install vLLM and dependencies
RUN pip3 install --no-cache-dir \
    vllm==0.4.2 \
    fastapi==0.110.0 \
    uvicorn==0.29.0 \
    pydantic==2.7.0

# Create app user
RUN useradd -m -u 1000 appuser
WORKDIR /app
COPY deployment/vllm_server.py /app/
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

ENTRYPOINT ["python3", "vllm_server.py"]
CMD ["--model", "Qwen/Qwen2.5-3B-Instruct", "--host", "0.0.0.0", "--port", "8000"]