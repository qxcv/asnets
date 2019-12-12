FROM ubuntu:19.10
RUN apt-get update -y && apt-get install -y python3-numpy python3-dev \
  python3-pip python3-wheel python3-venv flex bison build-essential \
  autoconf libtool git cmake libboost-all-dev
RUN useradd -ms /bin/bash asnets
COPY . /home/asnets/asnets
RUN chown -R asnets:asnets /home/asnets/asnets
USER asnets
WORKDIR /home/asnets/asnets
RUN python3 -m venv ${HOME}/venv-asnets
RUN . ${HOME}/venv-asnets/bin/activate \
  && echo . ${HOME}/venv-asnets/bin/activate >> ${HOME}/.bashrc \
  && pip install --upgrade pip \
  && pip install -e ${HOME}/asnets/
