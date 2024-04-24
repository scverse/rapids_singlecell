#!/bin/bash
docker_account=scverse
docker build -t rapids-singlecell:latest .
latest_id=$( docker images |grep -e "rapids-singlecell[ \t]*latest"|head -n1|awk '{print $3}' )
#docker tag $latest_id $docker_account/rapids-singlecell:latest
#docker push $docker_account/rapids-singlecell:latest