FROM python:3.11

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /app

COPY ./scripts/download_models.sh .

RUN ./download_models.sh

COPY ./requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./app .

CMD uvicorn src.main:app --host 0.0.0.0 --port 80 --proxy-headers

#CMD gunicorn src.main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:80 --proxy-protocol --timeout 60