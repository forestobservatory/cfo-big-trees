# cfo-big-trees

![Humboldt Redwoods State Park](https://salo.ai/assets/gallery/salo-forests-03.jpg)

**Saving the big ones.**

---

- [Introduction](#introduction)
- [Project partners](#project-partners)
- [Repository tools](#repository-tools)
    - [Conda](#conda)
    - [Pre-commit](#pre-commit)
- [Contact](#contact)

---

# Background

Landscapes throughout the western US are undergoing rapid changes in the face of global climate change and are impacting our reliance on the ecosystem services they provide. 

Managers, collaboratives, and other stakeholders are faced with difficult decisions regarding how best to manage for multiple resource benefits while fostering the development of resilient landscapes, capable of enduring future disturbances and ongoing climate change.

Large trees, historically a common feature, formed the backbone of western landscapes and played a keystone role in terrestrial and aquatic processes. Their thick bark, decay resistant wood, and massive size buffered them against fire and drought disturbances, while locking in carbon reserves and providing important habitat. These properties also made them a target for logging and their abundance is now a fraction of their pre Euro-American era levels. 

Currently, little information exists as to the distribution of large trees across the Sierran landscape, nor about the factors most associated with their persistence on the land. 

The goals of this project are to: 

- Identify the locations of large trees
- Quantify their functional value for carbon storage and wildlife
- Use machine learning to model their biophysical environment and predict likely locations for large trees in the future, and
- Use a landscape simulation model to determine the effects of treatments on the distribution of large trees across a 1 million hectare subregion and assess their variability under climate change.

This repository tracks our progress towards these goals.

---

# Project partners

The **Big Trees** project is a collaboration between [Blue Forest Conservation](https://www.blueforest.org/), [USFS Region 5](https://www.fs.usda.gov/r5) & [Salo Sciences](https://salo.ai).

---

# Repository tools

This repository contains scripts, figures & documentation for the **Big Trees** project. You can clone the repository locally with:

```bash
git clone https://github.com/forestobservatory/cfo-big-trees.git
```

Because we're working with a mix of collaborators and an array of data types, we store code from multiple languages:

- `python/` - contains standalone python scripts, typically for raw data processing.
- `R/` - contains standalone R scripts, typically for modeling and making figures.
- `notebooks/` - contains jupyter notebooks, likewise for modeling and making figures.

To facilitate collaboration across the team, we manage project dependencies through `conda`. 

## conda

`conda` manages package, dependencies & environments for many languages. I recommend installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html), which is available for Windows, Linux & Mac.

You can create the custom `big-trees` conda environment from the base repository directory with:

```bash
conda env update
```

This will install all the project dependencies into a standalone environment, which can be activated with:

```bash
conda activate big-trees
```

## pre-commit

[pre-commit](https://pre-commit.com/) is tool for managing and maintaining multi-language code standards and `git` hygiene. These utilities will ensure that all code has been subjected to a series of automated checks - like consistently formatting code or preventing personal tokens from being uploaded - before any code gets committed to the repository.

Once you've created and activated the `big-trees` conda environment, you can install the project's pre-commit hooks from the base directory of the repository:

```bash
pre-commit install
```

Once you've installed these hooks, `pre-commit` will run the automated code checks every time you run `git add {some-file.R}` and `git commit -m '{some message}'`. If it fails - don't worry! Pre-commit will typically apply the required changes needed to get the code properly formatted. You'll just need to re-run the above commands to add/commit the file again.

The hooks used by this project are managed by the `.pre-commit-config.yml` file. This includes a set of pre-commit hooks for python and [pre-commit hooks for R](https://www.rdocumentation.org/packages/precommithooks), which are mostly for linting and styling files.

---

# Contact

- Christopher Anderson (Salo Sciences) - [email](mailto:cba@salo.ai)
- Kim Quesnel (Blue Forest Conservation) - [email](mailto:kim@blueforest.org)
- Nicholas Povak (USFS Region 5) - [email](mailto:nicholas.povak@usda.gov)
