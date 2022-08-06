# Perfect Is the Enemy of Test Oracle

[![Install](https://img.shields.io/badge/Install-Instructions-blue)](INSTALL.md)
[![GitHub](https://img.shields.io/github/license/Intelligent-CAT-Lab/SEER)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6969272.svg)](https://doi.org/10.5281/zenodo.6969272)

Artifact repository for the paper _Perfect Is the Enemy of Test Oracle_, accepted at _ESEC/FSE 2022_.
Authors are [Ali Reza Ibrahimzada][ali], [Yiğit Varlı][yigit], [Dilara Tekinoğlu][dilara], and [Reyhaneh Jabbarvand][reyhaneh].

[ali]: https://alibrahimzada.github.io/
[yigit]: https://github.com/yigitv4rli
[dilara]: https://dtekinoglu.github.io/
[reyhaneh]: https://reyhaneh.cs.illinois.edu/index.htm

## Data Archive
Please find the model checkpoints and final datasets for both phase-1 and phase-2 of model training on [Zenodo](https://doi.org/10.5281/zenodo.6969272). Below is a description for each file of our data archive on Zenodo.
* `attention_analysis.zip`: This zip contains the necessary files for attention analysis. Please refer to [attention_analysis](attention_analysis) for further explanations.

## SEER Overview
Automation of test oracles is one of the most challenging facets of software testing, but remains comparatively less addressed compared to automated test input generation. Test oracles rely on a ground-truth that can distinguish between the correct and buggy behavior to determine whether a test fails (detects a bug) or passes. What makes the oracle problem challenging and undecidable is the assumption that the ground-truth should know the exact expected, correct or buggy behavior. However, we argue that one can still build an accurate oracle without knowing the exact correct or buggy behavior, but how these two might differ. This paper presents SEER, a Deep Learning-based approach that in the absence of test assertions or other types of oracle, can automatically determine whether a unit test passes or fails on a given method under test (MUT). To build the ground-truth, SEER jointly embeds unit tests and the implementation of MUTs into a unified vector space, in such a way that the neural representation of tests are similar to that of MUTs they pass on them, but dissimilar to MUTs they fail on them. The classifier built on top of this vector representation serves as the oracle to generate “fail” labels, when test inputs detect a bug in MUT or “pass” labels, otherwise. Our extensive experiments on applying SEER to more than 5K unit tests from a diverse set of opensource Java projects show that the produced oracle is (1) effective in predicting the fail or pass labels, achieving an overall accuracy, precision, recall, and F1 measure of 93%, 86%, 94%, and 90%, (2) generalizable, predicting the labels for the unit test of projects that were not in training or validation set with negligible performance drop, and (3) efficient, detecting the existence of bugs in only 6.5 milliseconds on average. Moreover, by interpreting the proposed neural model and looking at it beyond a closed-box solution, we confirm that the oracle is valid, i.e., it predicts the labels through learning relevant features.

## Implementation Details
Please check the following descriptions related to each .py file in this directory:
* `configs.py`: This file is used as a configuration when training deep learning models. We control hyperparameters, dataset, etc. from this file.
* `utils.py`: This file contains independent functions which are used throughout the project.
  * For creating the vocabulary for both phase-1 and phase-2, please execute the following:  
  `python3 utils.py create_vocabulary all`
  * For creating the .h5 files from JSONs, please execute the following if JSON type is train, fold is 1, and model type is JointEmbedder:  
  `python3 utils.py json_to_h5 train 1 JointEmbedder`
  * For extracting the length of the longest sequence (Code, Test) in phase-1 dataset, please execute the following:  
  `python3 utils.py get_max_len phase1_dataset_final`
  * For splitting the raw dataset in phase-1 into train (90%), valid (5%), and test sets (5%), please execute the following:  
  `python3 utils.py train_valid_test_split phase1_dataset_final 0.05 0.05`
  * For filtering the assert statements in the dataset, please execute the following:  
  `python3 utils.py filter_asserts`

The directory structure of SEER is as follows. Please read the corresponding README file inside each directory, if necessary.

     SEER
       |
       |--- attention_analysis:       The Multi-Head Attention Analysis related to model interpretation
       |
       |--- embedding_analysis:       The Embedding Analysis related to model interpretation
       |
       |--- dataset_generation:       The module which contains all scripts related to dataset generation
       |
       |--- models:                   The module which contains the deep learning models implemented in PyTorch
       |
       |--- learning:                 The module which contains everything related to model training and evaluation
       |
       |--- mutant_generation:        The module which contains all scripts related to mutant generation
           |
           |--- mutation_operators:   A directory which contains all mutation operators used in Major

## Contact

Please don't hesitate to open issues or pull-requests, or to contact us directly (alirezai@illinois.edu). We are thankful for any questions, constructive criticism, or interest. :blush:
