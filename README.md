# NTSX: Convert travel plans to graphs

## Installation

To install ntsx, we recommend using the [mamba](https://mamba.readthedocs.io/en/latest/index.html) package manager:

```
git clone git@github.com:fredshone/ntsx.git
cd ntsx
mamba create -n ntsx -c conda-forge -c city-modelling-lab -c pytorch --file requirements/base.txt --file requirements/dev.txt
mamba activate ntsx
pip install --no-deps -e .
```

### Jupyter Notebooks

To run the example notebooks you will need to add a ipython kernel into the mamba environemnt: `ipython kernel install --user --name=ntsx`.
