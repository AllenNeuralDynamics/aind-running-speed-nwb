# hash:sha256:10e80c554b19607edc482f73ec6510cd996ddf05977fada49fe1717ecd6df17d
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    h5py==3.11.0 \
    hdmf-zarr==0.8.0 \
    pandas==2.2.2 \
    pynwb==2.8.0 \
    scipy==1.13.1
