FROM tensorflow/tensorflow:latest-gpu

WORKDIR /home/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN apt install libgl1 -y
RUN apt install graphviz -y
