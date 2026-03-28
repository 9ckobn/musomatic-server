FROM python:3.12-alpine

ARG PROXY_URL=""
ENV PROXY_URL=${PROXY_URL}

RUN apk add --no-cache ffmpeg

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY engine.py server.py recommender.py ./

EXPOSE 8844
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8844"]
