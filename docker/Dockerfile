ARG PYTORCH="1.5"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6  libgl1-mesa-glx swig\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install mmdetection
RUN conda clean --all
RUN conda clean --all
RUN git clone --single-branch --branch DSV2_Baseline_FasterRCNN https://github.com/tuggeluk/mmdetection.git /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install --no-cache-dir -e .
RUN pip install cython --no-cache-dir
RUN pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
RUN pip install flask
RUN pip install git+https://github.com/yvan674/obb_anns
