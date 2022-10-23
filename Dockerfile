FROM jupyter/tensorflow-notebook
#FROM civisanalytics/datascience-python:latest

SHELL [ "rm", "-rf", "/app" ]

WORKDIR /app

COPY . .

ENTRYPOINT [ "python", "main.py" ]