FROM quay.io/pypa/manylinux_2_28_aarch64

RUN yum -y install dnf-plugins-core && \
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/aarch64/cuda-rhel8.repo && \
    yum -y clean all && yum -y makecache && \
    yum -y install \
      cuda-cudart-12-9 \
      cuda-cudart-devel-12-9 \
      libcublas-12-9 \
      libcublas-devel-12-9 \
      libcusparse-12-9 \
      libcusparse-devel-12-9 && \
    yum clean all

ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
