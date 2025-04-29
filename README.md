![DailyCIbadge](https://github.com/fred-shone/ntsx/actions/workflows/daily-ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/fredshone/ntsx/graph/badge.svg?token=QP2BI2J2CE)](https://codecov.io/gh/fredshone/ntsx)


# NTSX: Convert travel plans to graphs

Represent activity/trip chains as graphs.

1. Load NTS trips, individuals and households data
2. Embed as NTS as graphs with nodes representing activity locations (facilities) and edges representing trips
3. Process to represent repeat facility visits
4. Embed graphs and labels for Torch Geoemtric
5. Modelling tasks
    - Graph labelling
    - Node/edge labelling

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

### Embedding Experiment Idea

We compare a novel graph representation of activity sequences to established sequential and tabular representations. We consider three types of tasks: (i) graph labelling, (ii) edge/node labelling, and (iii) generation. Our work progresss understanding of how best to approach complex behavioural modelling problems using ML.

Compare representations:
- Tabular
- Sequence (discrete)
- Sequence (continuous)
- Graph

At various tasks:
- Graph labelling i.e. "does this person look employed based on their schedule?"
- Edge labelling i.e. "what mode/time/duration will be used for these trips?"
- Auto-regressive generation i.e. "what/when/where will be then next trip and activity?"

### Dev

![Coverage](https://codecov.io/gh/fredshone/ntsx/graphs/tree.svg?token=QP2BI2J2CE)
