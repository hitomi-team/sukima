FROM python:3.9.6
ENV PYTHONUNBUFFERED 1

WORKDIR /sukima
COPY requirements.txt /sukima/requirements.txt
RUN pip install -r requirements.txt
COPY ./app /sukima/app
