FROM python:3.9.6
ENV PYTHONUNBUFFERED 1

ENV PYTHONPATH "${PYTHONPATH}:/"
ENV PORT=8000

ENV PATH /usr/local/cuda/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/lib64
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

RUN mkdir /sukima
WORKDIR /sukima

COPY . /sukima
RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install --no-cache-dir -r requirements.txt
