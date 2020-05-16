FROM ubuntu:18.04
MAINTAINER Daniel Juravski

#RUN mkdir -p /root/heMoji/model
COPY docker/ heMoji/
COPY setup.py heMoji/
COPY lib/ heMoji/lib/
COPY src/ heMoji/src/


RUN apt-get update && apt-get install -y python2.7 python-pip nano
RUN cd heMoji && pip install -e .




ADD docker/README.sh /usr/local/bin
RUN chmod +x /usr/local/bin/README.sh
RUN echo "sh /usr/local/bin/README.sh" >> /root/.bashrc



ENTRYPOINT /bin/bash


# use setup in the root dir
# set ENTRYPOINT to cd /heMoji
