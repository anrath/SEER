# Insight into SEER

## Reproducing original paper's results
Our code is based on a fork of the SEER replication package. To read about their files, please consult [SEER_README.md](SEER_README.md). 

## Reproducing our paper's results
* `real_data_gen/`- This folder contains files related to testing SEER on new data.
* To execute our scripts, run [setup.sh](setup.sh). Note that this file executes a slurm script at the end. The department servers will complete this execution in approximately 2.5 hours.
* For creating the tables used in the paper execute `python real_data_gen/project_similarity_analysis.py` as well as the cells in the jupyter notebook `latex.ipynb`.
