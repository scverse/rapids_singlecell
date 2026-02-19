#!/bin/bash
set -euxo pipefail

docker_account=scverse
rapids_version=26.02

declare -A cuda_versions=(
    [cuda12]="12.8.0"
    [cuda13]="13.0.2"
)

declare -A cuda_archs=(
    [cuda12]="75-real;80-real;86-real;89-real;90"
    [cuda13]="75-real;80-real;86-real;89-real;90-real;100-real;120"
)

for label in cuda12 cuda13; do
    ver=${cuda_versions[$label]}

    grep -v -- '- rapids-singlecell' conda/rsc_rapids_${rapids_version}_${label}.yml > docker/rsc_rapids.yml
    docker build \
        --build-arg CUDA_VER="${ver}" \
        -t rapids-singlecell-deps:latest-${label} \
        -f docker/Dockerfile.deps \
        docker/
    rm docker/rsc_rapids.yml

    docker build \
        --build-arg CUDA_ARCHS="${cuda_archs[$label]}" \
        --build-context rapids-singlecell-deps=docker-image://rapids-singlecell-deps:latest-${label} \
        -t rapids-singlecell:latest-${label} \
        -f docker/Dockerfile \
        docker/
done

docker image ls -a
