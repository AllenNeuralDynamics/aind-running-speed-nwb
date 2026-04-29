# hash:sha256:0e89b72904321bc8969913348b919de11ec7e7408f4c895d20b83abd6a8b433c
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

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

ADD "https://github.com/coder/code-server/releases/download/v4.100.3/code-server-4.100.3-linux-amd64.tar.gz" /.code-server/code-server.tar.gz

RUN cd /.code-server \
    && tar -xvf code-server.tar.gz \
    && rm code-server.tar.gz \
    && ln -s /.code-server/code-server-4.100.3-linux-amd64/bin/code-server /usr/bin/code-server \
    && mkdir -p /.vscode/extensions

RUN code-server --extensions-dir="/.vscode/extensions" --install-extension ms-python.python@2025.14.0 \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension ms-toolsai.jupyter@2025.4.1 \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension reditorsupport.r@2.8.6 \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension saoudrizwan.claude-dev@3.35.1
