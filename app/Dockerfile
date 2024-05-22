# 
FROM python:3.8

WORKDIR /

COPY ./requirements.txt /requirements.txt

RUN pip install --no-cache-dir --upgrade -r /requirements.txt

COPY . /app

EXPOSE 3000

CMD ["uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "3000"]