ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 swig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN conda clean --all
RUN pip install cython

# Install OBB_Anns
WORKDIR /
RUN git clone https://github.com/yvan674/obb_anns.git
WORKDIR /obb_anns
#RUN python setup.py develop
RUN pip install -e .

# Install MMDetection deps
WORKDIR /
RUN git clone https://github.com/tuggeluk/mmdetection.git dummdet
ENV FORCE_CUDA="1"
RUN pip install -r /dummdet/requirements.txt
RUN pip install wandb
#RUN pip install --no-cache-dir -e .

# Install MMCV
#RUN pip install mmcv-full==latest+torch1.6.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
WORKDIR /
RUN git clone https://github.com/tuggeluk/mmcv.git
WORKDIR /mmcv
#ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib"
ENV MMCV_WITH_OPS=1
RUN pip install -e .

# Make mountpoint for disk sharing
# /mmdetection -> code base
# /workspace -> DS2 dataset
WORKDIR /mmdetection/data
RUN ln -s /workspace ds2_dense

# Setup ssh port forwarding
RUN apt-get update && apt-get install -y openssh-server
ADD .ssh.tar.gz /tmp/
RUN mkdir -p /root/.ssh \
        && mv /tmp/.ssh/authorized_keys /root/.ssh/authorized_keys \
        && rm -rf /tmp/.ssh .ssh.tar.gz \
        && chown root:root /root/.ssh/authorized_keys
RUN mkdir /var/run/sshd
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

# cd /mmdetection
# /opt/conda/bin/pip install -e .
# /opt/conda/bin/pip install future tensorboard
