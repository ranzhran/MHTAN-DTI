# MHTAN-DTI

This repository provides a reference implementation of MHTAN-DTI as described in this paper:
> MHTAN-DTI: Metapath-based Hierarchical Transformer and Attention Network for Drug-Target Interaction Prediction<br>
> Ran Zhang, Zhanjie Wang, Xuezhi Wang, Zhen Meng, Wenjuan Cui.<br>


## Dependencies

Recent versions of the following packages for Python 3.7 are required:
* PyTorch 1.12.1
* DGL 0.6.1
* NetworkX 2.6.3
* NumPy 1.21.6
* Scikit-learn 1.0.2


## Dataset

The preprocessed Luo et al. dataset is available at -> https://cstr.cn/31253.11.sciencedb.01726
* Luo et al. dataset contains 708 drugs, 1512 proteins, 5603 diseases and 4192 side effects.

## Usage

1. Create `checkpoint/` and `DTI_data/` directories
2. Download the dataset from the section above to `DTI_data/`
3. Execute the following command from the project home directory to train and test:
    * `python run_DTI.py`
