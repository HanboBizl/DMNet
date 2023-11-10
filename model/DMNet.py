import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import time
import os
import cv2
from skimage import io
import model.resnet as models
import model.vgg as vgg_models
from torchvision import transforms
from utils import util
from model.ASPP import ASPP
from model.ProtoContrastModule import PrototypeContrastLoss
##掩码平均池化GAP
def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005  #目标对象区域
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat
def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4

class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()
        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.layers = args.layers
        self.classes =args.classes
        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
        self.vgg = args.vgg
        self.dataset = args.data_set
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.BCEloss = nn.BCELoss()
        self.ppm_scales = args.ppm_scales
        self.fuse = args.fuse_select
        self.start_loss_iter = args.start_loss_iter
        from torch.nn import BatchNorm2d as BatchNorm
        self.pretrained = True
        self.classes = 2
        assert self.classes > 1
        assert self.layers in [50, 101, 152]

        if self.vgg:
            print('INFO: Using VGG_16 bn')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=self.pretrained)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
            self.layer3, self.layer4 = get_vgg16_layer(vgg16)

        else:
            print('INFO: Using ResNet {}'.format(self.layers))
            if args.layers== 50:
                resnet = models.resnet50(pretrained=self.pretrained)
            elif args.layers == 101:
                resnet = models.resnet101(pretrained=self.pretrained)
            else:
                resnet = models.resnet152(pretrained=self.pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                        resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        reduce_dim = 256
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512
        if self.dataset == 'iSAID':
            self.KMS_index = 11
        else:
            self.KMS_index = 5
        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1)
        )

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        ##1*1卷积，将2block特征与3block特征融合
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        mask_add_num = 1

        self.ASPP =ASPP(reduce_dim)
        self.init_merge = nn.Sequential(
                nn.Conv2d(reduce_dim * 2 + mask_add_num*3, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),)
        self.beta_conv =nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True))


        # 将4层特征图进行1*1卷积融合
        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim*5, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.coatten_linear = nn.Linear(256,256,bias=False)

        self.fuse_conv = nn.Conv2d(reduce_dim,reduce_dim,kernel_size=1,stride=1,padding=0,bias=False)
        self.coatten_linear_channel = nn.Linear(1024, 1024, bias=False)
        self.Sigmoid = nn.Sigmoid()
        init_value = 0.5
        self.afha1 = nn.Parameter(torch.FloatTensor([init_value]))
        self.afha2 = nn.Parameter(torch.FloatTensor([init_value]))
        self.beta1 = nn.Parameter(torch.FloatTensor([init_value]))
        self.beta2 = nn.Parameter(torch.FloatTensor([init_value]))
        self.contrast_loss = PrototypeContrastLoss()
        self.extra_gate = nn.Conv2d(reduce_dim,1,kernel_size=1,stride=1,padding=0,bias=False)

    def get_optim(self, model, args, LR):
        optimizer = torch.optim.SGD(
            [
                {'params': model.down_query.parameters()},
                {'params': model.down_supp.parameters()},
                {'params': model.init_merge.parameters()},
                {'params': model.alpha_conv.parameters()},
                {'params': model.beta_conv.parameters()},
                {'params': model.inner_cls.parameters()},
                {'params': model.res1.parameters()},
                {'params': model.res2.parameters()},
                {'params': model.ASPP.parameters()},
                {'params': model.extra_gate.parameters()},
                {'params': model.beta1},
                {'params': model.beta2},
                {'params': model.afha1},
                {'params': model.afha2},
                {'params': model.fuse_conv.parameters()},
                {'params': model.coatten_linear.parameters()},
                {'params': model.coatten_linear_channel.parameters()},
                {'params': model.cls.parameters()}],

            lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        return optimizer

    def freeze_modules(self, model):
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False

    # que_img, sup_img, sup_mask, que_mask(meta), que_mask(base), cat_idx(meta)
    def forward(self, x, s_x=torch.FloatTensor(1, 1, 3, 473, 473), s_y=torch.FloatTensor(1, 1, 473, 473),y=None,classes=None,proto_dict=None,bp_proto_dict=None,current_iter=None,start_loss=None):
        x_size = x.size()
        index=0
        if self.dataset == 'iSAID':
            h = int((x_size[2]) / 8 * self.zoom_factor)
            w = int((x_size[3]) / 8 * self.zoom_factor)
        else:
            h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
            w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        #   Query Feature
        label_pred = torch.cat(classes, 0).clone()  # bs*k

        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = self.layer4(query_feat_3)
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2), query_feat_3.size(3)),
                                             mode='bilinear', align_corners=True)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)


        #   Support Feature
        supp_feat_list = []
        supp_feat_fp_list = []
        supp_feat_bp_list = []
        final_supp_list = []
        feature_q_list = []
        mask_list = []
        q_map_list = []
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:, i, :, :, :])  # image：[batch,k,3,473,473] --> [batch,3,473,473]
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear',
                                     align_corners=True)
                supp_feat_4 = self.layer4(supp_feat_3 * mask)
                supp_feat_4_ori = self.layer4(supp_feat_3)  
                final_supp_list.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2), supp_feat_3.size(3)),
                                                mode='bilinear', align_corners=True)

            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_supp(supp_feat)
            bp_mask = F.relu(1-mask)
            supp_feat_fp = Weighted_GAP(supp_feat_4_ori, mask)
            supp_feat_bp = Weighted_GAP(supp_feat_4_ori, bp_mask)
            supp_feat_fp_list.append(supp_feat_fp)
            supp_feat_bp_list.append(supp_feat_bp)
            query_feat_attn, supp_feat_attn,q_map= self.attention_fuse_module(query_feat, supp_feat) #bs,c,h,w

            feature_q_list.append(query_feat_attn)
            q_map_list.append(q_map)

            supp_feat = Weighted_GAP(supp_feat_attn, mask)
            supp_feat_list.append(supp_feat)

        corr_query_mask_list = []
        cosine_eps = 1e-7

        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
            q = query_feat_4
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]

            tmp_query = q
            tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s
            tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)  # [batch,C,HW]
            tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)  # [batch,HW,C]
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

            similarity = torch.bmm(tmp_supp, tmp_query) / (
                        torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
            similarity = similarity.max(1)[0].view(bsize, sp_sz * sp_sz)
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                        similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)  # [bacth,1,H,W]
            corr_query = F.interpolate(corr_query, size=(query_feat_3.size()[2], query_feat_3.size()[3]),
                                       mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)
        corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)
        corr_query_mask = F.interpolate(corr_query_mask, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear',
                                        align_corners=True)

        if self.shot > 1:
            q_map = q_map_list[0]
            supp_feat = supp_feat_list[0]
            query_feat_attn = feature_q_list[0]
            for i in range(1, len(supp_feat_list)):
                supp_feat = supp_feat + supp_feat_list[i]
            supp_feat =supp_feat / len(supp_feat_list)
            # affinity_loss /= len(supp_feat_list)
            for i in range(1, len(feature_q_list)):
                query_feat_attn = query_feat_attn+feature_q_list[i]
            query_feat_attn = query_feat_attn / len(feature_q_list)
            for i in range(1, len(q_map_list)):
                q_map = q_map+ q_map_list[i]
            q_map =q_map/ len(q_map_list)

        if self.training:
            if current_iter >= start_loss:
                lead_proto_dict = {}  # {'0',1*256;'1'<1*256,.......}
                bp_lead_proto_dict = {}  # {'0',1*256;'1'<1*256,.......}

                for key in proto_dict.keys():
                    lead_proto_dict[key] = proto_dict[key].view(query_feat_4.shape[1], 1, 1)  # 256*1*1
                    bp_lead_proto_dict[key] = bp_proto_dict[key].view(query_feat_4.shape[1],1, 1)  # 256*1*1
                lead_map = []
                bp_list = []
                for feat,cls in zip(query_feat_4,label_pred): #256,32,32  -- 1
                    bp_sim = F.cosine_similarity(feat,bp_lead_proto_dict[cls.item()],dim=0).unsqueeze(0)  #1,32,32
                    bp_list.append(bp_sim.unsqueeze(0)) #1,1,32,32
                bp_map = torch.cat(bp_list,dim=0)   #8,1,32,32
                for key in sorted(lead_proto_dict.keys()):
                    similarity_map = F.cosine_similarity(query_feat_4, lead_proto_dict[key].unsqueeze(0),
                                                         dim=1).unsqueeze(
                        1)  # 8,1,32,32
                    lead_map.append(similarity_map)
                class_map = torch.cat(lead_map, dim=1)  # 8,10,32,32

                class_map = torch.cat([class_map,bp_map],dim=1)  #8,11,32,32
                class_mask = class_map.max(1)[1]  # 8,32,32
                class_prob = class_map.max(1)[0].unsqueeze(1)    #8,1,32,32
                for label, cls in zip(class_mask, label_pred): #256,32,32 --32,32 --1
                    label[label == cls.item()] = 1e-5
                    label[label != 1e-5] = 0
                    label[label == 1e-5] = 1
                class_mask = class_mask.unsqueeze(1)  # 8,1,32,32
                class_prob = class_prob*class_mask    # 8,1,32,32

            else:
                class_prob =torch.zeros(query_feat_4.shape[0],1,query_feat_4.shape[2],query_feat_4.shape[3]).cuda()
        else:
            lead_proto_dict = {}  # {'0',1*256;'1'<1*256,.......}
            # query_feat:1,256,32,32
            for key in proto_dict.keys():
                lead_proto_dict[key] = proto_dict[key].view(query_feat_4.shape[1], 1, 1)  # 256*1*1
            lead_map = []
            for key in sorted(lead_proto_dict.keys()):
                similarity_map = (F.cosine_similarity(query_feat_4, lead_proto_dict[key].unsqueeze(0),
                                                     dim=1).unsqueeze(1))  # 1,1,32,32
                lead_map.append(similarity_map)
            class_map = torch.cat(lead_map, dim=1)  # 1,10,32,32
            class_map[class_map<=0.5]=0  # 1,10,32,32

            val_similarity_map = (F.cosine_similarity(query_feat_4, supp_feat_fp, dim=1).unsqueeze(1))  # 1,1,32,32
            val_similarity_bp_map = F.cosine_similarity(query_feat_4, supp_feat_bp, dim=1).unsqueeze(1)  # 1,1,32,32

            class_map = torch.cat([class_map,val_similarity_bp_map,val_similarity_map], dim=1)  # 1,3,32,32
            class_mask = class_map.max(1)[1]  # 1,32,32s
            class_prob = class_map.max(1)[0].unsqueeze(1)  # 1,1,32,32

            class_mask[class_mask == self.KMS_index] = 1e-5
            class_mask[class_mask != 1e-5] = 0
            class_mask[class_mask == 1e-5] = 1
            class_mask = class_mask.unsqueeze(1)  # 1,1,32,32
            class_prob = class_prob * class_mask  # 8,1,32,32

        bin = int(query_feat_attn.shape[2])
        query_feat_bin = nn.AdaptiveAvgPool2d(bin)(query_feat_attn)
        supp_feat_bin = supp_feat.expand(-1, -1, bin, bin)
        corr_mask_bin = F.interpolate(corr_query_mask, size=(bin, bin), mode='bilinear',
                                      align_corners=True)  # 将先验掩码resize
        q_map = F.interpolate(q_map, size=(bin, bin), mode='bilinear',
                                align_corners=True)

        merge_feat_bin = torch.cat([query_feat_bin, supp_feat_bin, corr_mask_bin, q_map, class_prob], 1)
        merge_feat_bin = self.init_merge(merge_feat_bin)

        merge_feat_bin = self.beta_conv(merge_feat_bin) + merge_feat_bin
        merge_feat = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)),
                                           mode='bilinear', align_corners=True)

        merge_feat = self.ASPP(merge_feat)
        merge_feat = self.res1(merge_feat)
        merge_feat = self.res2(merge_feat) + merge_feat
        out_0 = self.cls(merge_feat)
        pred1 = out_0.softmax(1)


        ### CSRM
        Sim_FP, Sim_BP = self.CSRM_Sim(query_feat, out_0)

        if self.fuse:
            Sim_FP_1,Sim_BP_1=self.Filter(query_feat,pred1,Sim_FP,Sim_BP)
            out_1 = self.similarity_func(query_feat, Sim_FP_1, Sim_BP_1)
        else:
            out_1 = self.similarity_func(query_feat, Sim_FP, Sim_BP)

        #   Output Part
        if self.zoom_factor != 1:
            out = F.interpolate(out_1, size=(h, w), mode='bilinear', align_corners=True)
            out_0=F.interpolate(out_0, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            main_loss = self.criterion(out, y.long())
            aux_loss  = self.criterion(out_0,y.long())

            proto_dict ,bp_proto_dict= self.contrast_loss(supp_feat_fp_list,supp_feat_bp_list,classes, proto_dict,bp_proto_dict)

            return out.max(1)[1], main_loss, aux_loss, proto_dict,bp_proto_dict
        else:
            return out

    def CSRM_Sim(self, feature_q, out):
        bs = feature_q.shape[0]  # 1,256,60,60
        pred_1 = out.softmax(1)  # 1,2,60,60
        pred_1 = pred_1.view(bs, 2, -1)  # bs，2，h*w:1,2,3600
        pred_fg = pred_1[:, 1]  # bs，1，h*w:1,1,3600
        pred_bg = pred_1[:, 0]  # bs，1，h*w:1,1,3600
        fg_ls = []
        bg_ls = []
        for epi in range(bs):
            fg_thres = 0.7  # 0.9 #0.6
            bg_thres = 0.6  # 0.6
            cur_feat = feature_q[epi].view(feature_q.shape[1], -1)  # 256，h*w
            f_h, f_w = feature_q[epi].shape[-2:]
            if (pred_fg[epi] > fg_thres).sum() > 0:
                fg_feat = cur_feat[:, (pred_fg[epi] > fg_thres)]  # .mean(-1)   #1024,N1

            else:
                fg_feat = cur_feat[:, torch.topk(pred_fg[epi], 12).indices]  # .mean(-1)
            if (pred_bg[epi] > bg_thres).sum() > 0:
                bg_feat = cur_feat[:, (pred_bg[epi] > bg_thres)]  # .mean(-1)   #1024,N2
            else:
                bg_feat = cur_feat[:, torch.topk(pred_bg[epi], 12).indices]  # .mean(-1)

            fg_proto = fg_feat.mean(-1)
            bg_proto = bg_feat.mean(-1)
            fg_ls.append(fg_proto.unsqueeze(0))
            bg_ls.append(bg_proto.unsqueeze(0))
        new_fg = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)  # k*c>>>k*c*1*1
        new_bg = torch.cat(bg_ls, 0).unsqueeze(-1).unsqueeze(-1)  # k*c*1*1
        return new_fg, new_bg

    def similarity_func(self, feature_q, fg_proto, bg_proto):
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)

        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        return out

    def masked_average_pooling(self,feature,mask):
        masked_feature=torch.sum(feature*mask,dim=(2,3))/(mask.sum(dim=(2,3))+1e-5)
        masked_feature=masked_feature.unsqueeze(-1).unsqueeze(-1)
        return masked_feature
    def Filter(self,query_feat,pred1,Sim_FP,Sim_BP):
        thresh_fp = 0.7
        thresh_bp = 0.6
        pred_fp = pred1[:, 1, :, :]  # b*60*60
        pred_bp = pred1[:, 0, :, :]  # b*60*60
        fp_mask = (pred_fp >= thresh_fp).float().unsqueeze(1)  # b*1*60*60
        bp_mask = (pred_bp >= thresh_bp).float().unsqueeze(1)  # b*1*60*60
        fulse_mask = ((pred_fp < thresh_fp) & (pred_bp < thresh_bp)).float().unsqueeze(1)  # b*1*60*60
        fulse_feat = query_feat * fulse_mask  # b*c*60*60

        fp_proto = Sim_FP
        bp_proto = Sim_BP
        for x in range(2):
            fulse_result = self.similarity_func(fulse_feat, fp_proto, bp_proto).softmax(1)  # b*2*60*60
            fulse_fp_mask = (fulse_result[:, 1, :, :] > thresh_fp - 0.05).float().unsqueeze(1)
            fulse_bp_mask = (fulse_result[:, 0, :, :] > thresh_bp - 0.02).float().unsqueeze(1)
            fp_mask = fp_mask + fulse_fp_mask
            bp_mask = bp_mask + fulse_bp_mask
            fulse_mask = 1 - fp_mask - bp_mask  # b*1*60*60
            fulse_feat = query_feat * fulse_mask
            fp_proto = self.masked_average_pooling(query_feat, fp_mask)  # b*c*1*1
            bp_proto = self.masked_average_pooling(query_feat, bp_mask)  # b*c*1*1
        fulse_result = self.similarity_func(fulse_feat, fp_proto, bp_proto).softmax(1)  # b*2*60*60
        fulse_fp_mask = (fulse_result.max(1)[1]).float().unsqueeze(1)
        fulse_bp_mask = (fulse_result.min(1)[1]).float().unsqueeze(1)
        fp_mask = fp_mask + fulse_fp_mask
        bp_mask = bp_mask + fulse_bp_mask
        fp_proto = self.masked_average_pooling(query_feat, fp_mask)  # b*c*1*1
        bp_proto = self.masked_average_pooling(query_feat, bp_mask)  # b*c*1*1

        Sim_FP_1 = Sim_FP * 0.1 + fp_proto * 0.9
        Sim_BP_1 = Sim_BP * 0.3 + bp_proto * 0.7
        return Sim_FP_1,Sim_BP_1


    def attention_fuse_module(self,feature_q,feature_s):
        # bs,256,60,60
        bs, c, h, w = feature_q.shape[:]
        feature_q_flat = feature_q.view(bs, c, -1)  # bs,c,hw
        feature_s_flat = feature_s.view(bs, c, -1)  # bs,c,hw
        feature_q_t = torch.transpose(feature_q_flat, 1, 2).contiguous()  # bs,hw,c

        q_att = self.coatten_linear(feature_q_t)  # bs,hw,c
        feature_mul = torch.bmm(q_att, feature_s_flat)  # bs,hw,hw,获得相似图
        s_map = feature_mul.softmax(1)
        q_map = torch.transpose(feature_mul.softmax(2), 1, 2)
        feature_s_att = torch.bmm(feature_q_flat, s_map).contiguous()  # support的互注意力机制图,bs,c,hw
        feature_q_att = torch.bmm(feature_s_flat, q_map).contiguous()  # query的互注意力机制图, bs,c,hw
        feature_s_att =  self.beta1* feature_s_att.view(bs, c, h, w)  # bs,c,h,w

        feature_q_att = 0.5*feature_q + self.afha1* feature_q_att.view(bs, c, h, w)  # bs,c,h,w
        feature_q_flat_channel = feature_q_flat #bs,c,hw
        feature_s_flat_channel = torch.transpose(feature_s_flat,1,2)    #bs,hw,c

        feature_mul_channel = torch.bmm(self.coatten_linear_channel(feature_q_flat_channel),feature_s_flat_channel).contiguous() #bs,c,c
        s_channel_map = torch.transpose(feature_mul_channel.softmax(1),1,2)
        q_channel_map = feature_mul_channel.softmax(2)
        feature_s_channel_att = torch.bmm(s_channel_map, feature_q_flat_channel).contiguous()    #bs,c,hw
        feature_q_channel_att = torch.bmm(q_channel_map,feature_s_flat).contiguous()             #bs,c,hw
        feature_s_channel_att =  self.beta2*feature_s_channel_att.view(bs,c,h,w)

        feature_q_channel_att = 0.5*feature_q + self.afha2*feature_q_channel_att.view(bs,c,h,w)

        feature_q_out = self.fuse_conv(feature_q_att + feature_q_channel_att)
        feature_s_out = self.fuse_conv(feature_s_att + feature_s_channel_att)   #bs,c,h,w


        q_map = self.Sigmoid(feature_mul.mean(2)).view(bs, 1, h, w)


        return  feature_q_out, feature_s_out ,q_map



