FROM python:3.9.6
ENV PYTHONUNBUFFERED 1

RUN mkdir /gpt-server
WORKDIR /gpt-server

COPY requirements.txt /gpt-server
RUN pip install -r requirements.txt

COPY . /gpt-server