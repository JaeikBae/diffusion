FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel

# Install dependencies
RUN apt update && apt install -y \
    vim \
    git \
    wget 

# Install Python dependencies
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /ws