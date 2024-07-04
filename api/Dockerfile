FROM python:3.8

WORKDIR /

COPY . .

RUN apt-get -y update
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install -r requirements.txt

EXPOSE 3000

CMD gunicorn --bind 0.0.0.0:3000 main:app --timeout 600