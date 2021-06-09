packages <- c(
  "tidyverse",
  "ggpmisc",
  "stringr",
  "raster",
  "docopt",
  "lintr",
  "remotes",
  "git2r",
  "styler",
  "VGAM"
)

repo_url <- "https://ftp.osuosl.org/pub/cran"

install.packages(packages, repos = repo_url)
remotes::install_github("lorenzwalthert/precommit@v0.1.3.9010")
