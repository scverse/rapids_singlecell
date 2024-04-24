FROM condaforge/miniforge3
SHELL ["/bin/bash", "-c"]
COPY ../conda/rsc_rapids_24.04.yml rsc_rapids_24.04.yml
RUN set -x \
    && apt update \
    && apt -y dist-upgrade \
    && apt -y autoremove \
    && apt -y clean \
    && mamba env create -p /rsc -f rsc_rapids_24.04.yml \
    && /rsc/bin/python -c "import rapids_singlecell as rsc" \
    && mamba install -p /rsc pytest -c conda-forge \
    && mamba clean -ay

WORKDIR /rsc/bin

ENTRYPOINT ["/rsc/bin/python"]
