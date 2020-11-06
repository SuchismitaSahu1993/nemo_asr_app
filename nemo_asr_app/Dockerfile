################################################################################
# Application Container
################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

################################################################################
# NGC base image
#-------------------------------------------------------------------------------
#FROM nvcr.io/nvidia/nemo:v0.11
#FROM nvcr.io/nvidia/pytorch:20.08-py3
#FROM nvcr.io/nvidia/pytorch:20.07-py3
FROM nvcr.io/nvidia/nemo:v1.0.0b1

################################################################################
# Install system modules
#-------------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update

# Install EasyDict for manifest
RUN pip install easydict wordcloud

# Install Cython
#RUN pip install Cython

# NeMo and wandb install
RUN pip install wandb
#RUN pip install nemo_toolkit[asr]==0.10.1
#RUN pip install nemo_toolkit[asr]==0.11
#RUN pip install git+https://github.com/NVIDIA/NeMo.git@candidate#egg=nemo_toolkit[all]
#RUN pip install nemo_toolkit[all]==1.0.0b1
#RUN docker pull nvcr.io/nvidia/nemo:v1.0.0b1

# Install nodejs for jupyterlab
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -
RUN apt-get install -y nodejs

# install jupyter lab extensions
RUN pip install jupyterlab-nvdashboard
RUN jupyter labextension install jupyterlab-nvdashboard
RUN jupyter lab build

# MCV sox processing
RUN apt-get install -y libsox-fmt-mp3
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg

# NeMo ASR application
#COPY . /home/ssahu/nemo_asr_app
#COPY lab /root/.jupyter/lab
################################################################################
ENV APP_DIR /home/ssahu/nemo_asr_app

ENV PATH=$PATH:/usr/src/tensorrt/bin
ENV PYTHONPATH "${PYTHONPATH}:/tmp/NeMo/:${APP_DIR}"

# Expose Jupyter & Tensorboard
EXPOSE 8888
EXPOSE 6006

# /workspace contains NVIDIA tutorials and example code
WORKDIR /home/ssahu/nemo_asr_app

# Start Jupyter up by default rather than a shell
ENTRYPOINT ["/bin/sh"]
CMD ["-c", "jupyter lab  --notebook-dir=/home/ssahu/nemo_asr_app --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"]
