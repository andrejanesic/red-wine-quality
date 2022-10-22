FROM jupyter/tensorflow-notebook
#FROM civisanalytics/datascience-python:latest

SHELL [ "rm", "-rf", "/app" ]

WORKDIR /app

COPY src .

ENTRYPOINT [ "python", "main.py" ]