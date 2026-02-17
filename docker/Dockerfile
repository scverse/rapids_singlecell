FROM rapids-singlecell-deps

ARG GIT_ID=main

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ENV PATH=/opt/conda/bin:$PATH
ENV SKBUILD_CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=75-real;80-real;86-real;89-real;90-real;100-real;120"

RUN <<EOF
# install rapids_singlecell from source (compiled for all supported GPU architectures)
set -x
mkdir /src
cd /src
git clone https://github.com/scverse/rapids_singlecell.git
cd rapids_singlecell
git checkout ${GIT_ID}
/opt/conda/bin/python -m pip install --no-cache-dir -e .
EOF
