#!/bin/sh
PARTITION=Segmentation



GPU_ID=0

dataset=iSAID # iSAID/ LoveDA

exp_name=split0

arch=DMNet
visualize=False

net=resnet50 # vgg resnet50

exp_dir=exp/DMNet_iSAID/${dataset}/${arch}/${exp_name}/${net}
snapshot_dir=${exp_dir}/snapshot
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}_resnet50_DMNet.yaml
mkdir -p ${snapshot_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp test.sh test.py ${config} ${exp_dir}

echo ${arch}
echo ${config}
echo ${visualize}

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u test.py \
        --config=${config} \
        --arch=${arch} \
        --visualize =${visualize} \
        2>&1 | tee ${result_dir}/test-$now.log