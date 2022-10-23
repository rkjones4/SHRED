import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('methods/')
import os
import numpy as np
from pn_loader import *

bceloss = nn.BCEWithLogitsLoss()

class PNPP_PointEnc(nn.Module):
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        
        return xyz, features
    
    def __init__(self, input_channels):        
        super(PNPP_PointEnc, self).__init__()

        use_xyz = True
        use_bn = False
        
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[input_channels, 32, 32, 64],
                use_xyz=use_xyz,
                bn = use_bn
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=use_xyz,
                bn = use_bn
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=use_xyz,
                bn = use_bn
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=use_xyz,
                bn = use_bn
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(
            PointnetFPModule(mlp=[128 + input_channels, 128, 128, 128], bn = use_bn)
        )
        self.FP_modules.append(
            PointnetFPModule(mlp=[256 + 64, 256, 128], bn = use_bn)            
        )
        self.FP_modules.append(
            PointnetFPModule(mlp=[256 + 128, 256, 256], bn = use_bn)            
        )
        self.FP_modules.append(
            PointnetFPModule(mlp=[512 + 256, 256, 256], bn = use_bn)            
        )
        
            
    def forward(self, pointcloud):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_features[0].transpose(1,2)

        
class DMLP(nn.Module):
    def __init__(self, ind, hdim1, hdim2, odim, DP):
        super(DMLP, self).__init__()
        
        self.l1 = nn.Linear(ind, hdim1)
        self.l2 = nn.Linear(hdim1, hdim2)
        self.l3 = nn.Linear(hdim2, odim)
        self.d1 = nn.Dropout(p=DP)
        self.d2 = nn.Dropout(p=DP)
        
    def forward(self, x):
        x = self.d1(F.relu(self.l1(x)))
        x = self.d2(F.relu(self.l2(x)))
        return self.l3(x)


class FixNet(nn.Module):
    def __init__(self, DP = 0.0):
        super(FixNet, self).__init__()
        
        self.pn_enc = PNPP_PointEnc(4)
        self.pt_head = DMLP(128, 64, 32, 1, DP)
        
    def forward(self, x):
                
        enc = self.pn_enc(x)
        pred = self.pt_head(enc)        

        return pred.squeeze(-1)

    def loss(self, x, gt):
        loss = bceloss(x, gt)
        with torch.no_grad():
            acc = ((x >= 0.0) == gt.bool()).float().mean().item()

        return loss, acc
        
def load_model(args):
    net = FixNet(        
        DP=args.dropout,
    )
            
    net.args = args
    return net
