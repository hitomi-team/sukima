FROM python:3.9.6
ENV PYTHONUNBUFFERED 1

ENV PYTHONPATH "${PYTHONPATH}:/"
ENV PORT=8000

RUN mkdir /sukima
WORKDIR /sukima

COPY . /sukima
RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install --no-cache-dir -r requirements.txt
