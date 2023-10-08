# HermesSim

This repository contains the code and the dataset for our USENIX Security '24 paper:

> Haojie He, Xingwei Lin, Ziang Weng,
Ruijie Zhao, Shuitao Gan, Libo Chen, Yuede Ji, Jiashui Wang, and Zhi Xue. *Code is not Natural Language: Unlock the Power of Semantics-Oriented Graph Representation for Binary Code Similarity Detection*. USENIX Security '24.

## Artifacts

Description of folders: 

- lifting: contains helper scripts for lifting binary functions into Pcode based graphs. 
- preprocess: contains scripts for graph normalization and encoding. 
- model: contains the neural network model and related experiments configures. 
- postprocess: contains scripts for testing pairs generation, fast evaluation and visualization. 
- binaries: contains raw binaries of the datasets. 
    - For Dataset-1 published by [the previous work](https://www.usenix.org/system/files/sec22-marcelli.pdf), please refer [Binaries](https://github.com/Cisco-Talos/binary_function_similarity/tree/main/Binaries). 
    - Binaries of Dataset-RTOS are available at [here](#TODO). 
- bin: binaries of external tools. Currently, only there is [gsat-1.0.jar](#TODO). 
- dbs: contains description files and feature files (including the extracted graphs) of the datasets. Available at [here](#TODO). 
- inputs: contains the inputs for the neural network models (the outputs of the preprocessing step). Available at [here](#TODO). 
- outputs: contains the outputs of the neural network models (checkpoint files, inferred embeddings, log and configure files, and etc.) and the outputs of fast evaluation (`summary_*.csv` and `*_MRR_Recall_max.csv` files). Available at [here](#TODO). 


## How to reproduce the experiments

### 1. Lifting binary functions Pcode based representations. 

- Related folders: lifting, dbs, bin, binaries

> If you are only interested in running experiments on the two datasets used in the paper, you can skip this step since all the results you need are in the `dbs' folder.

First, constuct CFG summary files:
```sh
python lifting/dataset_summary.py \
    --cfg_summary dbs/Dataset-1/cfg_summary/testing \
    --dataset_info_csv dbs/Dataset-1/testing_Dataset-1.csv \
    --cfgs_folder dbs/Dataset-1/features/testing/acfg_features_Dataset-1_testing
```

This script takes the following files as inputs: 
1. a dataset description csv file that contains the indices of all functions in the dataset; 
2. and a folder of files that contain the CFGs of functions;

You can refer [binary_function_similarity](https://github.com/Cisco-Talos/binary_function_similarity) to figure out how to generate these artifacts for customized datasets. 

Then, lifting binaries using:
```sh
python lifting/pcode_lifter.py \
    --cfg_summary ./dbs/Dataset-1/cfg_summary/testing \
    --output_dir ./dbs/Dataset-1/features/testing/pcode_raw_Dataset-1_testing \
    --graph_type ALL \
    --verbose 1
```

This script takes the generated CFG summary files and binaries as input. It will invoke the [GSAT](https://github.com/sgfvamll/gsat) executable to conduct the major work. 

### 2. Preprocess inputs

- Related folders: preprocess, dbs, inputs

The second step is graph normalization and encoding. 
See `preprocess/preprocess_all.sh` for examples. 


### 3. Model Training / Inferring

- Related folders: model, inputs, outputs

The following example will run the representation part of our ablation study, including both training and inferring. The inferring step will output the embeddings of all functions in the testing dataset. More configure files can be found in `model/configures`. 

```sh
CUDA_VISIBLE_DEVICES=1 python model/main.py \
    --inputdir dbs \
    --config ./model/configures/e02_repr.json \
    --dataset=one
```

By default, the `model/main.py` will put results in the `outputs` folder. 


### 4. Result Analysis

- Related folders: postprocess, outputs

4.1 Generate Testing Pairs

This step samples testing pairs from the whole testing dataset. The following example generates testing pairs for the XM task with 1000 query functions and 10000 negative functions per query. The script outputs a `pos-*.csv` file and a `neg-*.csv` file, which contain postives pairs and negative pairs, respectively. 

``` python
python postprocess/1.generate_testing/testing_generator.py \
    dbs/Dataset-1/pairs/experiments/full \
    postprocess/1.generate_testing/configures/full/xm-1000-10000_Ds1.json
```

All sampling configure files can be found at `postprocess/1.generate_testing/configures`. 


4.2 Summarize Results

This step evaluate the results with generated `pos-*.csv` files, `neg-*.csv` files, and embeddings of the testing dataset.  

Example of evaluating the x64-XO task with pool size 100: 
```python
python postprocess/2.summarize_results/collect_stats.py \
    dbs/Dataset-1/pairs/experiments/x64/pos-xo-1000-100-arch_x-bit_64_Ds1.csv \
    outputs/experiments/
```

Example of evaluating all tasks (may consume a lot of time): 
```python
python postprocess/2.summarize_results/collect_stats.py \
    dbs/Dataset-1/pairs/experiments/ \
    outputs/experiments/
```

4.3 Print or plot results

Please see `postprocess/3.pp_results` for details. 

- `postprocess/3.pp_results/print_plot_results.ipynb`: Results of the comparative experiments and the ablation study in the paper. 
- `postprocess/3.pp_results/print_rtos_results.py`: Results of the real-world vulnerability search experiments. 
- `postprocess/3.pp_results/appendix-c.ipynb`: Results in the Appendix-C. 


<!-- ## How to cite our work -->

## Feedback

If you need help or find any bugs, feel free to submit GitHub issues or PRs. 

