# hash:sha256:f8ddd95e9dd1353c3aa1ac17cf7faa3d7b9e0dc257b8470e8fd942901305666f
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

ENV VERSION=1.0

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.4.0 \
    aind-nwb-utils==0.1.0 \
    h5py==3.11.0 \
    hdmf-zarr==0.11.0 \
    pandas==2.2.2 \
    pynwb==2.8.3 \
    scipy==1.13.1
