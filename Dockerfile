# SatQuery AI — Phase 1 sovereign runtime (SIH26167 / ISRO SAC).
# Built on a connected host, then docker save/load onto the air-gapped GPU node.
# Target: Ubuntu 22.04, Python 3.10, GDAL 3.8.4, CUDA 12.1 runtime.

# ---------------------------------------------------------------------------
# Stage 1: compile GDAL 3.8.4 and install pinned Python wheels into a venv
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ARG GDAL_VERSION=3.8.4
ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=4

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        wget \
        git \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        python3 \
        python3-dev \
        python3-venv \
        python3-pip \
        python3-distutils \
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
        libxerces-c-dev \
        libspatialite-dev \
        libpq-dev \
        sqlite3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python3 --version \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

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

ENV CPLUS_INCLUDE_PATH=/usr/local/include \
    C_INCLUDE_PATH=/usr/local/include \
    GDAL_CONFIG=/usr/local/bin/gdal-config \
    GDAL_DATA=/usr/local/share/gdal \
    PROJ_LIB=/usr/share/proj \
    LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN pip install --upgrade pip setuptools wheel

# CUDA 12.1 wheels first so a later requirements pass does not replace them with CPU builds.
RUN pip install torch==2.2.1 torchvision==0.17.1 \
        --index-url https://download.pytorch.org/whl/cu121

WORKDIR /build
COPY requirements.txt /build/requirements.txt
# Keep the cu121 torch build. Use PyPI wheels for rasterio/shapely so the
# image does not require a from-source Cython compile against GDAL/GEOS.
RUN grep -vE '^(torch|torchvision)(=|$)' /build/requirements.txt > /tmp/req-py.txt \
    && pip install -r /tmp/req-py.txt

# ---------------------------------------------------------------------------
# Stage 2: slim CUDA runtime, non-root process user
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GDAL_CONFIG=/usr/local/bin/gdal-config \
    GDAL_DATA=/usr/local/share/gdal \
    PROJ_LIB=/usr/share/proj \
    CPLUS_INCLUDE_PATH=/usr/local/include \
    C_INCLUDE_PATH=/usr/local/include \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu \
    PYTHONPATH=/app \
    LOCAL_MODELS_DIR=/local_models \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-distutils \
        libproj22 \
        libgeos-c1v5 \
        libspatialindex6 \
        libtiff5 \
        libgeotiff5 \
        libpng16-16 \
        libjpeg-turbo8 \
        libopenjp2-7 \
        libwebp7 \
        libsqlite3-0 \
        libcurl4 \
        libxml2 \
        libexpat1 \
        libxerces-c3.2 \
        libspatialite7 \
        libpq5 \
        proj-data \
        libzstd1 \
        liblzma5 \
        libdeflate0 \
        libjson-c5 \
        curl \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy GDAL artifacts only — do not overlay the CUDA toolkit under /usr/local.
COPY --from=builder /usr/local/lib/libgdal.so* /usr/local/lib/
COPY --from=builder /usr/local/share/gdal /usr/local/share/gdal
COPY --from=builder /usr/local/bin/gdal-config /usr/local/bin/gdal-config
COPY --from=builder /opt/venv /opt/venv
RUN ldconfig && gdal-config --version

WORKDIR /app

COPY backend /app/backend
COPY models /app/models
COPY scripts /app/scripts

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin satquery \
    && mkdir -p \
        /app/data/raw \
        /app/data/processed \
        /local_models/sam \
        /local_models/bigearthnet \
        /local_models/cdvqa \
        /local_models/vllm \
    && chown -R satquery:satquery /app /local_models /opt/venv \
    && chmod +x /app/scripts/init_env.sh

USER satquery

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=10s --retries=8 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
