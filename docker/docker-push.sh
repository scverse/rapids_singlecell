#!/bin/bash
set -euxo pipefail

docker_account=scverse
rapids_version=24.06
grep -v -- '- rapids-singlecell' conda/rsc_rapids_${rapids_version}.yml > rsc_rapids.yml
docker build -t rapids-singlecell-deps:latest -f docker/Dockerfile.deps .
rm rsc_rapids.yml
docker build -t rapids-singlecell:latest -f docker/Dockerfile .
latest_id=$( docker images |grep -e "rapids-singlecell[ \t]*latest"|head -n1|awk '{print $3}' )
#docker tag $latest_id $docker_account/rapids-singlecell:latest
#docker push $docker_account/rapids-singlecell:latest
