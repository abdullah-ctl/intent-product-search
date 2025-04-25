FROM python:3.10-slim

WORKDIR /app

COPY finetune-service/requirements.txt .
RUN pip install -r requirements.txt

# Instead of copying everything, copy only the needed files
COPY finetune-service/finetune.py .

CMD ["python", "finetune.py"]
