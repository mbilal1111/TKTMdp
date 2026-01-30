# TKTMdp

Reproducibility repository for **TKTMdp**, an ontology-grounded action admissibility mechanism for deep reinforcement learning (DRL) baseload optimisation in energy communities.

## What this repo contains
- `knowledge/`: ontology, RDF/triples, and expert-labelled state–action artefacts used to build the admissibility function \(F(s,a)\) and valid action set \(A_v(s)\)
- `src/`: training, evaluation, and analysis code
- `configs/`: experiment configurations (DDPG/PPO baseline vs TKTMdp)
- `paper/`: LaTeX sources
- `results/`: generated tables/figures

## Quickstart (reproduce tables)
This repo supports a “fast reproduce” path that regenerates the evaluation table from the provided aggregated results CSV.

### Create environment (Conda)
```bash
conda env create -f environment.yml
conda activate tktmdp
