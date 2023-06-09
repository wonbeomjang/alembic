FROM pytorch/pytorch:latest

RUN apt-get update
RUN apt-get -y install libglib2.0-0
RUN apt-get -y install libgl1-mesa-glx gcc

WORKDIR /app/
COPY ./requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt