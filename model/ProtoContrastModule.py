from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask.float(), (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


class PrototypeContrastLoss(nn.Module, ABC):
    def __init__(self):
        super(PrototypeContrastLoss, self).__init__()

    def _class_construct(self, pros, labels_):  # 0,0,0,1,2,3,4
        unique_labels = torch.unique(labels_)  # 得到bs个的标签类别 0,1,2,3,4
        for i in range(len(pros)):
            pros[i]=pros[i].unsqueeze(1)    #8,1,256,1,1
        pros = torch.cat(pros,dim=1) # bs,k,c,1,1
        pros = pros.view(-1,pros.shape[-3],pros.shape[-2],pros.shape[-1]) #bs*k,c,1,1----8,256,1,1
        pro_dict = dict()
        for i in range(len(unique_labels)): #0
            index = torch.where(labels_ == unique_labels[i])    #

            pro_dict[unique_labels[i].item()] = pros[index].contiguous().view(-1, 2048)  # {0:Tensor(1,256),1:,,,,2:,,,}


        return pro_dict

    def forward(self, s_fp_list, s_bp_list, classes, proto_dict,bp_proto_dict):
        # q_fp:bs*c*1*1 , s_fp_list:5*bs*c*1*1, classes:(k,k,k),  proto_dict:dict()
        classes = torch.cat(classes,0).clone() #将(bs,k)展开成1纬bs*k的张量

        class_dict = self._class_construct(s_fp_list, classes)
        class_bp_dict = self._class_construct(s_bp_list, classes)
        for key in class_dict.keys():
            if key not in proto_dict:
                proto_dict[key] = torch.mean(class_dict[key],dim=0,keepdim=True).detach()
            else:
                orignal_value = proto_dict[key] #1,2048
                proto_dict[key] = torch.mean(torch.cat([orignal_value,class_dict[key]],dim=0),dim=0,keepdim=True).detach()
        for key in class_bp_dict.keys():
            if key not in bp_proto_dict:
                bp_proto_dict[key] = torch.mean(class_bp_dict[key],dim=0,keepdim=True).detach()
            else:
                orignal_value = bp_proto_dict[key]
                bp_proto_dict[key] = torch.mean(torch.cat([orignal_value,class_bp_dict[key]],dim=0),dim=0,keepdim=True).detach()

        return  proto_dict,bp_proto_dict