# Start with Pytorch pre-installed.
# This image comes with built-in CUDA support as well.
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

# General image-level setup.
SHELL ["/bin/bash", "-c"]
WORKDIR /root/sasn

RUN \
# Install a text editor.
    apt-get update && \
    apt-get install -y emacs && \
# Install NTLK data.
    pip install --no-cache-dir nltk && \
    python -c "import nltk; nltk.download('punkt')"

# Copy datasets.
COPY datasets/ /root/sasn/datasets

# Copy SASN source.
COPY sasn/ /root/sasn
