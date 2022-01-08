FROM python:3.9.6
ENV PYTHONUNBUFFERED 1

ENV PYTHONPATH "${PYTHONPATH}:/"
ENV PORT=8000

RUN mkdir /sukima
WORKDIR /sukima

COPY . /sukima
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN alembic upgrade head
