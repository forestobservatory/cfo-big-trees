FROM continuumio/miniconda:latest

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

# install the conda environment
COPY environment.yml .
RUN conda update -n base -c defaults conda -y \
  && conda env create --file environment.yml && conda clean --all -y

# container entry command
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "big-trees"]
