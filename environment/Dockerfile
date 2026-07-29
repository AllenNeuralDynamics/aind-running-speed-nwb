# hash:sha256:a78d90f3c8a9e878bfdcf26a3e8f4ddbe52c397c6b853a0cb5cfd3061abd7338
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:code-server4.129.0python3.12.4codeocean-mcp-serveruvpyreflyetc

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

ENV VERSION=1.0

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.4.0 \
    aind-nwb-utils==0.1.0 \
    h5py==3.11.0 \
    hdmf-zarr==0.12.0 \
    matplotlib==3.10.9 \
    pandas==2.2.2 \
    pynwb==3.1.3 \
    scipy==1.13.1
