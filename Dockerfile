FROM ubuntu:18.04
LABEL maintainer="Daniel Juravski"


RUN mkdir -p /root/heMoji/
WORKDIR /root/heMoji/

COPY docker/ .
COPY setup.py .
COPY lib/ lib/
COPY src/ src/
COPY datasets/he_sentiment_twitter/token/train.tsv \
    datasets/he_sentiment_twitter/token/dev.tsv \
    datasets/he_sentiment_twitter/token/test.tsv \
    data/amram_2017/
RUN mv examples.txt data/
RUN mv sentiment_model.hdf5 data/amram_2017/model.hdf5
RUN mv sentiment_examples.txt data/amram_2017/examples.txt


RUN apt-get update && apt-get install -y python2.7 python-pip nano
RUN pip install -e .


RUN mv README.sh /usr/local/bin/
RUN echo "/usr/local/bin/README.sh" >> /root/.bashrc


ENTRYPOINT /bin/bash
