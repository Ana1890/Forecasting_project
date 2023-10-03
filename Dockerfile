# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.9-slim

# WORKDIR /app
# COPY . /app

# # Install pip requirements
# COPY requirements.txt ./app
# RUN pip install -r .app/requirements.txt 
# # Inside the container 

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt 

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "model_file.py"]
