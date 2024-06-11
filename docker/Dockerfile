FROM rapids-singlecell-deps

ARG GIT_ID=main

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ENV PATH=/opt/conda/bin:$PATH

RUN <<EOF
# install rapids_singlecell from source
set -x
mkdir /src
cd /src
git clone https://github.com/scverse/rapids_singlecell.git
cd rapids_singlecell
git checkout ${GIT_ID}
/opt/conda/bin/python -m pip install --no-cache-dir -e .
EOF

ENTRYPOINT ["/opt/conda/bin/ipython"]
