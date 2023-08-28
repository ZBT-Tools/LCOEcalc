FROM python:3.11-slim AS build

RUN apt-get -y update
RUN apt-get -y install git

# set working directory in container
RUN mkdir wd
WORKDIR wd

# Copy and install packages
COPY requirements.txt .
#COPY .env .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY webapp/ ./webapp/

EXPOSE 8050

# Run locally
CMD gunicorn --bind 0.0.0.0:8050 --workers 4 webapp.app:server
# CMD gunicorn --bind 0.0.0.0:$PORT webapp:server
#CMD exec gunicorn --bind 0.0.0.0:2357 --workers 1 --timeout 0 webapp:server
