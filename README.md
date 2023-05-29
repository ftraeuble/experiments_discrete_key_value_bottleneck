## Discrete Key Value Bottleneck Codebase 


> [**Discrete Key-Value Bottleneck**](https://arxiv.org/abs/2207.11240)
> *Frederik Träuble, Anirudh Goyal, Nasim Rahaman, Michael Mozer, Kenji Kawaguchi, Yoshua Bengio, Bernhard Schölkopf*. ICML 2023.

### 1. Prerequisites

To reproduce the results of the paper, you need to first create 
a conda environment and install the package:

```bash   
conda create -n kvb python=3.10.6
conda activate kvb 
git clone git@github.com:ftraeuble/experiments_discrete_key_value_bottleneck.git
cd experiments_discrete_key_value_bottleneck
pip install .
```

### 2. Reproduce the Experiments

To reproduce the toy experiments from Fig. 2 in the paper, you can run the following notebook:

- [Reproduce Experiments Figure 2](
experiments_discrete_key_value_bottleneck%2Fnotebooks%2Freproduce_experiments_figure_2.ipynb)

To reproduce the main experiments CIFAR10 you need to first log in to wandb in your machine, set your wandb 
`PROJECT_NAME` and `PROJECT_ENTITY` as well as the `PROJECT_ROOT_DIR` environment variable. 

```bash
export PROJECT_NAME=YOUR_PROJECT_NAME
export PROJECT_ENTITY=YOUR_PROJECT_ENTITY
export PROJECT_ROOT_DIR=YOUR_PROJECT_ROOT_DIR
```

To run the experiments it is advisable to precompute the relevant backbone embeddings across all required datasets. To reproduce the ConvMixer experiments, you will have to download 
the CIFAR10 and Imagenet32 Embeddings for the ConvMixer backbone from the SDMLP paper submission [repository](https://github.com/anon8371/AnonPaper1) from Bricken et al. (2023).

To precompute the embeddings, you can use the following two notebooks:

- [Precompute ConvMixer Embeddings](
experiments_discrete_key_value_bottleneck%2Fnotebooks%2Fcreate_convmixer_embeddings.ipynb)  

- [Precompute Other Backbone Embeddings](
experiments_discrete_key_value_bottleneck%2Fnotebooks%2Fcreate_backbone_embeddings.ipynb)

Finally, store all created embeddings and label files in a folder named `backbone_embeddings` within `PROJECT_ROOT_DIR`.
 
A list of all sweeps can be found in the directory `sweeps/icml2023`. Run the following command:

```bash
wandb sweep sweeps/icml2023/NAME_OF_SWEEP.yaml
wandb agent <SWEEP_ID>
```

All sweeps comprise 300+ trained models, which can be used to reproduce all results of the paper.


#### Single Experiment

A single model can be trained by running the following command:

```bash
python scripts/train.py --backbone=resnet50_imagenet_v2 --dim_key=14 --dim_value=10 --init_epochs=10 --learning_rate=0.3 --num_books=256 --num_pairs=4096 --pretrain_data=CIFAR100 --seed=2
```

### Cite us

If you found this codebase useful, please cite our paper:

```
@article{trauble2023discrete,
  title={Discrete Key-Value Bottleneck},
  author={Tr{\"a}uble, Frederik and Goyal, Anirudh and Rahaman, Nasim and Mozer, Michael and Kawaguchi, Kenji and Bengio, Yoshua and Sch{\"o}lkopf, Bernhard},
  journal={International Conference on Machine Learning},
  year={2023}
}
```
