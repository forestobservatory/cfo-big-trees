FROM ubuntu:20.04

SHELL ["/bin/bash", "-c"]

# create working directory
ENV APP_HOME /app
WORKDIR $APP_HOME

# install gsutil dependencies
RUN apt-get --allow-releaseinfo-change update -yq \
  && apt-get install -yq gcc python-dev python-setuptools libffi-dev curl git \
  && apt-get autoclean -y \
  && rm -rf /var/lib/apt/lists/*

# install gsutil
RUN curl -sSL https://sdk.cloud.google.com | bash

# install conda
ENV INSTALLER installer.sh
ENV CONDA_DIR ${APP_HOME}/miniconda3
ENV CONDA ${CONDA_DIR}/bin/conda
RUN curl -o ${INSTALLER} "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
  && bash ${INSTALLER} -b -p ${CONDA_DIR} \
  && rm ${INSTALLER}

# install conda env
COPY environment.yml .
RUN ${CONDA} update -n base -c defaults conda -y \
  && ${CONDA} env create --file environment.yml && ${CONDA} clean --all -y

ENV PATH="${CONDA_DIR}/bin:${PATH}"
