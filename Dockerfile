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
RUN /opt/conda update -c defaults conda -y \
  && /opt/conda env create --file environment.yml && /opt/conda clean --all -y

# container entry command
ENTRYPOINT ["/opt/conda", "run", "--no-capture-output", "-n", "big-trees"]
