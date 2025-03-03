FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /

COPY ./ /
COPY ./src/requirements.txt /src/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /src/requirements.txt

EXPOSE 8000

ENTRYPOINT ["python", "/src/main.py"]