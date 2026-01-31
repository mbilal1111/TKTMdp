# Data (SimBench)

This project uses SimBench dataset ID41:

- SimBench code: 1-LV-semiurb4--1-sw
- Scenario date: 2024
- Source page: https://simbench.de/en/download/datasets/

## How to obtain
Download the dataset from the SimBench datasets page (ID41). The website provides a CSV download (zip) containing time-series tables such as LoadProfile, RESProfile, PowerPlantProfile, and StorageProfile.

## Local paths expected by this repo
- data/raw/simbench_id41/         # extracted dataset files (from the zip)
- data/processed/                # generated/converted inputs for training (created by scripts)

## Notes
If browser downloads fail, try a different browser (SimBench website note).
