#!/bin/sh
PARTITION=Segmentation


GPU_ID=7
dataset=iSAID # iSAID/LoveDA

exp_name=split0

arch=DMNet

net=resnet50 # vgg resnet50

model=model/${arch}.py
Cocontrast=model/ProtoContrastModule.py
exp_dir=exp/DMNet_iSAID/${dataset}/${arch}/${exp_name}/${net}
snapshot_dir=${exp_dir}/snapshot
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}_resnet50_DMNet.yaml
mkdir -p ${snapshot_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp train.sh train.py  ${model}  ${Cocontrast} ${config} ${exp_dir}

echo ${arch}
echo ${config}

CUDA_VISIBLE_DEVICES=${GPU_ID} /usr/bin/python3 -m torch.distributed.launch --nproc_per_node=1 train.py \
        --config=${config} \
        --arch=${arch} \
        2>&1 | tee ${result_dir}/train-$now.log