# Insight into SEER

## Reproducing original paper's results
Our code is based on a fork of the SEER replication package. To read about their files, please consult [SEER_README.md](SEER_README.md). 

## Reproducing our paper's results
* `real_data_gen/`- This folder contains files related to testing SEER on new data.
* To execute our scripts, run [setup.sh](setup.sh). Note that this file executes a slurm script at the end. The department servers will complete this execution in approximately 2.5 hours.
* For creating the tables used in the paper execute `python real_data_gen/project_similarity_analysis.py` as well as the cells in the jupyter notebook `latex.ipynb`.
* To recreate attention analysis results:
  * To recreate the attention matrices for Phase 2 unseen data, first navigate to the `attention_analysis` directory with `cd attention_analysis`, then run `attention_analysis.py` with `python attention_analysis.py --model JointEmbedder --dataset TestOracleInferencePhase2 --gpu_id 0 --fold_number 1 --reload_from 29`. Our results can be found in `attention_analysis/attention_weights_images/` and `attention_analysis/attention_weights_matrices/`, and this is also where your results should generate.
  * To recreate the attention matrices for New Data, first navigate to the `attention_analysis` directory with `cd attention_analysis`, then run `attention_analysis_phase3.py` with `python attention_analysis_phase3.py --model JointEmbedder --dataset TestOracleInferencePhase2 --gpu_id 0 --fold_number 1 --reload_from 29`. Our results can be found in `attention_analysis/phase3_no_try_except/attention_weights_images_phase3/` and `attention_analysis/attention_weights_matrices_phase3/`, and this is also where your results should generate.
  * To recreate the attention analysis results for Phase 2 unseen data, first navigate to the `attention_analysis` directory with `cd attention_analysis`, then run `main.java` by compiling with `javac main.java` then running with `java main`. Our results can be found in `attention_analysis/phase2_main_analysis_results.txt`, and your results should generate in `attention_analysis/phase2_main_analysis.txt`.
  * To recreate the attention analysis results for New Data, first navigate to the `attention_analysis` directory with `cd attention_analysis`, then run `phase3_analysis_main.java` by compiling with `javac phase3_analysis_main.java` then running with `java phase3_analysis_main`. Our results can be found in `attention_analysis/phase3_main_analysis_results.txt`, and your results should generate in `attention_analysis/phase3_main_analysis.txt`.
