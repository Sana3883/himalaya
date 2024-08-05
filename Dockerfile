ROM continuumio/miniconda3:latest

WORKDIR /app

RUN conda install -y -c conda-forge git pip numpy scipy cython joblib pytest compilers cupy cudnn cutensor nccl


RUN apt-get update -y && \
    apt-get install -y gcc g++ libarchive-dev libarchive13

#RUN conda config --set solver_class classic


RUN pip install himalaya torch
