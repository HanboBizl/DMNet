# Not just Learning from Others but Relying on Yourself: A new perspective on Few-Shot Segmentation in Remote Sensing

This repository contains the source code for our paper "*Not just Learning from Others but Relying on Yourself: A new perspective on Few-Shot Segmentation in Remote Sensing*" by Hanbo Bi, Yingchao Feng, Zhiyuan Yan, Yongqiang Mao, Wenhui Diao, Hongqi Wang, Xian Sun.

> **Abstract:** *Few-shot segmentation (FSS) is proposed to segment unknown class targets with just a few annotated samples. Most current FSS methods follow the paradigm of mining the semantics from the support images to guide the query image segmentation. However, such a pattern of “learning from others” struggles to handle the extreme intraclass variation, preventing FSS from being directly generalized to remote sensing scenes. To bridge the gap of intraclass variance, we develop a dual-mining network named DMNet for cross-image mining and self-mining, meaning that it no longer focuses solely on support images but pays more attention to the query image itself. Specifically, we propose a class-public region mining (CPRM) module to effectively suppress irrelevant feature pollution by capturing the common semantics between the support–query image pair. The class-specific region mining (CSRM) module is then proposed to continuously mine the class-specific semantics of the query image itself in a “filtering” and “purifying” manner. In addition, to prevent the coexistence of multiple classes in remote sensing scenes from exacerbating the collapse of FSS generalization, we also propose a new known-class metasuppressor (KMS) module to suppress the activation of known-class objects in the sample. Extensive experiments on the iSAID and LoveDA remote sensing datasets have demonstrated that our method sets the state of the art with a minimum number of model parameters. Significantly, our model with the backbone of Resnet-50 achieves the mean Intersection over Union (mIoU) of 49.58% and 51.34% on iSAID under 1- and 5-shot settings, outperforming the state-of-the-art method by 1.8% and 1.12%, respectively.\.*

## Code Structure

```
├─DMNet
|   ├─test.py
|   ├─test.sh
|   ├─train.py
|   ├─train.sh
|   ├─util
|   ├─model
|   |   ├─ASPP.py
|   |   ├─DMNet.py
|   |   ├─resnet.py
|   |   ├─vgg.py
|   ├─lists
|   |   ├─iSAID
|   |   |   ├─fss_list
|   |   |   ├ train.txt
|   |   |   ├ val.txt
|   |   ├─LoveDA
|   |   |   ├─fss_list
|   |   |   ├ train.txt
|   |   |   ├ val.txt
|   ├─data
|   ├─config
|   |   ├─iSAID
|   |   ├─LoveDA
```

## Data Preparation

- Create a folder `data` at the same level as this repo in the root directory.

  ```
  cd ..
  mkdir data
  ```
- Download the iSAID and LoveDA dataset from our [[Baidu]](https://pan.baidu.com/s/1NjZxFxLCNcaTCu_uQO8NNA?pwd=2f3)(code: 2f3y) and put it in the `data` directory.
- Or you can follow the process as described in Section IV of the paper.

## Backbone Preparation

- Create a folder `backbones` at the same level as this repo in the root directory.
  ```
  cd ..
  mkdir backbones
  ```
- Download the backbones from our [[Baidu]](https://pan.baidu.com/s/1l9CPkmP69sbxzUYUtwVISg?pwd=n314)(code: n314) and put it in the `backbones` directory.
- 
## Getting Started

### Training 

Execute this command at the root directory: 

```
bash train.sh {*GPU_ID*} {*dataset*} {*exp_name*} {*arch*} {*net*}
```
### Testing

Execute this command at the root directory: 
```
bash test.sh {*GPU_ID*} {*dataset*} {*exp_name*} {*arch*} {*net*}
```

### Visualization

```
bash test.sh {*visualize=True*}
```
-Note the save path is defined in vis.py

## Related Repositories

-This project is built upon a very early version of **PFENet**: https://github.com/dvlab-research/PFENet and **BAM**: https://github.com/chunbolang/BAM. 

-Many thanks to their greak work!

## Features

- [x] Distributed training (Multi-GPU)
- [x] Different dataset divisions
- [x] Multiple runs

# Citation

If you find this project useful, please consider citing:
```
@article{bi2023not,
  title={Not just Learning from Others but Relying on Yourself: A new perspective on Few-Shot Segmentation in Remote Sensing},
  author={Bi, Hanbo and Feng, Yingchao and Yan, Zhiyuan and Mao, Yongqiang and Diao, Wenhui and Wang, Hongqi and Sun, Xian},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2023},
  publisher={IEEE}
}
```

