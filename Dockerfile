# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.9-slim

WORKDIR /mlflow

COPY . .

RUN pip install --no-cache-dir -r requirements.txt 

EXPOSE 5000
