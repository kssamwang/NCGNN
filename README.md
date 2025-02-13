# Clarify Confused Nodes via Separated Learning
In this paper, we first propose a new metric, termed Neighborhood Confusion (NC), to facilitate a more reliable separation of nodes. These pave the way for Neighborhood Confusion-guided Graph Convolutional Network (NCGCN), in which nodes are grouped by their NC values and accept intra-group weight sharing and message passing. 
### 1 Overview of Model Architecture and Performance

![](images/architecture.png)
![](images/experiment.png)

### 2 Python environment setup with Conda
```bash
conda env create -f requirement.yaml
```

### 3 Data Preparation
For all datasets, we adopt dense data splitting as in [20], i.e., randomly split them into training, validation and testing sets with a proportion of 60%/20%/20%.

### 4 Code Execution
#### 4.1 File Structure
- data -- put data in this dir
- model -- GCN(NCGCN) model
- index -- index of nodes' neighborhood
- main.py -- run this 
- utils.py -- utils

#### 4.2 Code Execution
```bash
python main.py --dataset computers --model NCGCN --device 3 --lr 5e-2 --wd 5e-5 --dp1 0.2 --dp2 0.6 --hops 2 --addself 0 --threshold 0.7 --hidden 512
```

#### 4.3 Hyper-parameter Space
- learning rate: [1e-3, 5e-3, 1e-2, 5e-2, 0.1]
- weight decay: [0, 5e-5, 1e-4, 5e-4, 1e-3]
- dropout rate: [0, 0.9] with 0.1 interval
- hop k: [1, 2]
- Threshold T: [3, 4, 5, 6, 7]
- Add self-loop or not: [True, False]

### 5 Citation
If you find this work useful, please cite our paper:
```
@article{zhou2025ncgnn,
  title={Clarify Confused Nodes via Separated Learning},
  author={Jiajun Zhou, Shengbo Gong, Xuanze Chen, Chenxuan Xie, Shanqing Yu, Qi Xuan and Xiaoniu Yang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025}
}
```