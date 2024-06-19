# hash:sha256:5e53a943dbe6417171708b9db680446798a1bc82cdcec16d5a381e336e3ba7a5
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    h5py \
    hdmf-zarr \
    pandas \
    pynwb \
    scipy
