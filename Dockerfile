FROM ubuntu:20.04

SHELL ["/bin/bash", "-c"]

# create working directory
ENV APP_HOME /app
WORKDIR $APP_HOME

# install gsutil dependencies
RUN apt-get --allow-releaseinfo-change update -yq \
  && apt-get install -yq gcc python-dev python-setuptools libffi-dev curl \
  && apt-get autoclean -y \
  && rm -rf /var/lib/apt/lists/*

# install gsutil
RUN curl -sSL https://sdk.cloud.google.com | bash

# install conda
ENV INSTALLER installer.sh
ENV CONDA_DIR=${APP_HOME}/miniconda3
RUN curl -o ${INSTALLER} "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
  && bash ${INSTALLER} -b -p ${CONDA_DIR} \
  && rm ${INSTALLER} \
  && ${CONDA_DIR}/bin/conda init \
  && source /root/.bashrc

# install the conda environment
COPY environment.yml .
RUN conda update -n base -c defaults conda -y \
  && conda env create --file environment.yml && conda clean --all -y

# container entry command
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "big-trees"]
