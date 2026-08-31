# SatQuery AI backend — CUDA 12.1 devel Ubuntu 22.04 with native GDAL C++ libs.
# Python GDAL/rasterio wheels compile against GDAL 3.8.4 headers (binary-compliant).
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ARG GDAL_VERSION=3.8.4
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONPATH=/app
ENV LOCAL_MODELS_DIR=/local_models
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=4

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3 \
    gdal-bin \
    libgdal-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libproj-dev \
    libgeos-dev \
    libgeos++-dev \
    libspatialindex-dev \
    libtiff-dev \
    libgeotiff-dev \
    libpng-dev \
    libjpeg-dev \
    libopenjp2-7-dev \
    libwebp-dev \
    zlib1g-dev \
    libsqlite3-dev \
    libcurl4-openssl-dev \
    libxml2-dev \
    libexpat1-dev \
    libpq-dev \
    sqlite3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Overlay GDAL 3.8.4 so pip GDAL==3.8.4 / rasterio==1.3.9 match the C++ ABI.
WORKDIR /tmp/gdal
RUN wget -q "https://github.com/OSGeo/gdal/releases/download/v${GDAL_VERSION}/gdal-${GDAL_VERSION}.tar.gz" \
    && tar -xzf "gdal-${GDAL_VERSION}.tar.gz" \
    && cmake -S "gdal-${GDAL_VERSION}" -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_TESTING=OFF \
        -DBUILD_PYTHON_BINDINGS=OFF \
        -DGDAL_USE_EXTERNAL_LIBS=ON \
    && cmake --build build \
    && cmake --install build \
    && ldconfig \
    && gdal-config --version \
    && rm -rf /tmp/gdal

ENV CPLUS_INCLUDE_PATH=/usr/local/include:/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/local/include:/usr/include/gdal
ENV GDAL_CONFIG=/usr/local/bin/gdal-config
ENV GDAL_DATA=/usr/local/share/gdal
ENV PROJ_LIB=/usr/share/proj
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --upgrade pip \
    && pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p \
        /local_models/sam \
        /local_models/bigearthnet \
        /local_models/cdvqa \
        /local_models/vllm \
        /app/models \
        /app/uploads \
        /app/artifacts

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
