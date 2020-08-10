FROM python:3.7.3-slim

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER author@sample.com
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt-get etc.
apt-get pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
## Do not edit if you have a requirements.txt
RUN pip install -r requirements.txt

