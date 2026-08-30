# Ubuntu 22.04 / Python 3.10 CPU fallback.
# Prefer the repository-root Dockerfile for CUDA 12.1 + GDAL 3.8.4 (source build).
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    LOCAL_MODELS_DIR=/local_models

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        gnupg \
        wget \
        curl \
        git \
        build-essential \
        python3 \
        python3-dev \
        python3-venv \
        python3-pip \
        libproj-dev \
        libgeos-dev \
        libspatialindex-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libopenjp2-7-dev \
        zlib1g-dev \
    && add-apt-repository -y ppa:ubuntugis/ubuntugis-unstable \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        gdal-bin \
        libgdal-dev \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python --version \
    && python -m pip install --upgrade pip \
    && gdal-config --version \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal \
    GDAL_DATA=/usr/share/gdal \
    PROJ_LIB=/usr/share/proj

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

COPY . /app
RUN mkdir -p /local_models/sam /local_models/bigearthnet /local_models/cdvqa /local_models/vllm \
    && useradd --create-home --uid 1000 --shell /usr/sbin/nologin satquery \
    && chown -R satquery:satquery /app /local_models

USER satquery

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
