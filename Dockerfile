FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/tmp

ENV GRADIO_ANALYTICS_ENABLED=False
ENV GRADIO_NO_UPGRADE_CHECK=True
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV DO_NOT_TRACK=1

CMD ["python", "app-local-models.py"]