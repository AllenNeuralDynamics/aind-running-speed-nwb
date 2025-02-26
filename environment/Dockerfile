# hash:sha256:4bf7050422418093056badbd1b9770f98aa6eca0f1182739ca221a172b1668f1
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        vim=2:8.1.2269-1ubuntu5.31 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    h5py==3.11.0 \
    hdmf-zarr==0.8.0 \
    pandas==2.2.2 \
    pynwb==2.8.0 \
    scipy==1.13.1
