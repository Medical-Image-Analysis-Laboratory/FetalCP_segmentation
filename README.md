# "Multi-dimensional topological loss for cortical plate segmentation in fetal brain MRI"


This repository contains the code used for the [paper](https://arxiv.org/): "Multi-dimensional topological loss for cortical plate segmentation in fetal brain MRI". 

Please cite the paper if you are using either our python implementation or model.

## Installation 
1) Clone the github repository.
2) Create a conda environment using the `environment.yml`.

## Usage

### 1. Update paths in ```config_paths.json```

### 2. Training

```python train.py [fold]```

### 3. Inference

Inference only:
```python infer_feta20.py```

With performance metrics computation:
```python infer_and_eval_feta20.py```
