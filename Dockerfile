FROM python:3.9.6
ENV PYTHONUNBUFFERED 1

ENV PYTHONPATH "${PYTHONPATH}:/"
ENV PORT=8000

WORKDIR /sukima

COPY requirements.txt /sukima/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
