FROM debian:stable-slim

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	ca-certificates git wget sudo build-essential libgl1-mesa-glx locales && \
  rm -rf /var/lib/apt/lists/*

ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} pmuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
ARG HOME=/home/pmuser
WORKDIR ${HOME}

USER pmuser
ENV MINICONDA_VERSION 4.7.12.1
ENV CONDA_DIR ${HOME}/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

COPY detectron2-env-cpu.yml ${HOME}/
RUN conda env create -f ${HOME}/detectron2-env-cpu.yml

COPY tlpim-env-cpu.yml ${HOME}
RUN conda env create -f ${HOME}/tlpim-env-cpu.yml

RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile && \
    /bin/bash -c "source ~/.profile" 
RUN conda init bash
RUN echo "conda activate detectron2"  >> ${HOME}/.bashrc

SHELL ["conda", "run", "-n", "detectron2", "/bin/bash", "-c"]
RUN pip install torch==1.7.0+cpu torchvision==0.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

RUN pip install 'git+https://github.com/facebookresearch/fvcore'
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
ENV FORCE_CUDA="0"
ENV TORCH_CUDA_ARCH_LIST=""
RUN pip install -e detectron2_repo

SHELL ["conda", "run", "-n", "tlpim", "/bin/bash", "-c"]
RUN pip install git+https://github.com/rcmalli/keras-vggface.git

USER root
RUN export PATH='/home/pmuser/miniconda3/bin:$PATH' \
  && echo $PATH \
  && echo '** cleaning caches...' \
  && conda clean --all -y \
  && /bin/rm -rf /home/pmuser/*.yml \
  && /bin/rm -rf /home/pmuser/.cache/pip \
  && echo '** cleaning caches done.'

USER pmuser
RUN echo "export PYTHONPATH=${HOME}/segmentation:${HOME}/recognition" >> ${HOME}/.bashrc
COPY models ${HOME}/models
COPY samples ${HOME}/samples
COPY segmentation ${HOME}/segmentation
COPY recognition ${HOME}/recognition

ENV FVCORE_CACHE="/tmp"

