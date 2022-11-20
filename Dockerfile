# Start with Pytorch pre-installed.
# This image comes with built-in CUDA support as well.
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

SHELL ["/bin/bash", "-c"]
WORKDIR /root/sasn

# Install a text editor.
RUN apt-get update && \
    apt-get install -y vim
# Install NLTK data.
RUN pip install --no-cache-dir nltk && \
    python -c "import nltk; nltk.download('punkt')"

COPY datasets/ /root/sasn/datasets
COPY sasn/ /root/sasn
