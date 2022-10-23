import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('methods/')
import os
import numpy as np
from pn_loader import *

bceloss = nn.BCEWithLogitsLoss(reduction = 'mean')

class PNPP_ShapeEnc(nn.Module):
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        
        return xyz, features
    
    def __init__(self, input_channels, use_bn):        
        super(PNPP_ShapeEnc, self).__init__()

        use_xyz = True
        
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
                mlp=[256, 256, 512, 1024],
                use_xyz=use_xyz,
                bn = use_bn
            )
        )        

        
                    
    def forward(self, pointcloud):

        xyz, features = self._break_up_pc(pointcloud)
        for i in range(len(self.SA_modules)):
            xyz, features = self.SA_modules[i](xyz, features)

        return features.squeeze(-1)        
        

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


class MergeNet(nn.Module):
    def __init__(self, use_bn, DP = 0.0):
        super(MergeNet, self).__init__()

        self.enc_net = PNPP_ShapeEnc(5, use_bn)
        self.head_net = DMLP(1024, 256, 64, 1, DP)

            
    def forward(self, x):        

        enc = self.enc_net(x)
        pred = self.head_net(enc)                
        return pred.squeeze(-1)
                
    def loss(self, x, gt):        

        loss = bceloss(x, gt)
                               
        with torch.no_grad():
            P = x >= 0.
            GT = gt == 1.0
            pos_corr = (P & GT).float().sum().item()
            res = {
                'corr': (P == GT).float().sum().item(),
                'total': gt.shape[0] * 1.,
                'pos_corr': pos_corr,
                'prec_denom': pos_corr + (P & ~GT).float().sum().item(),
                'rec_denom': pos_corr + (~P & GT).float().sum().item()                
            }
                                
        return loss, res

        
def load_model(args):
    net = MergeNet(
        use_bn = (args.use_bn == 'y'),
        DP=args.dropout,        
    )            
    net.args = args
    return net
        
                        
