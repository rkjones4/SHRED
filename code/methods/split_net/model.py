import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('methods/')
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from pn_loader import *

bceloss = nn.BCEWithLogitsLoss(reduction='none')
celoss = nn.CrossEntropyLoss()

MIN_REG_POINTS = 10
MIN_REG_PURITY = 0.5

class PNPP_PointEnc(nn.Module):
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        
        return xyz, features
    
    def __init__(self, input_channels, use_bn):        
        super(PNPP_PointEnc, self).__init__()

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


        
class DSMLP(nn.Module):
    def __init__(self, ind, hdim, odim, DP):
        super(DSMLP, self).__init__()
        
        self.l1 = nn.Linear(ind, hdim)
        self.l2 = nn.Linear(hdim, odim)
        self.d1 = nn.Dropout(p=DP)
                
    def forward(self, x):
        x = self.d1(F.relu(self.l1(x)))
        return self.l2(x)


class SplitNet(nn.Module):
    def __init__(self, num_clusters, DP, use_bn=True, match_mode='hung_os'):
        super(SplitNet, self).__init__()

        self.pn_enc = PNPP_PointEnc(3, use_bn)
        self.cluster_net = DSMLP(128, 64, num_clusters, DP)

        self.match_mode = match_mode
        # NOTE
        # HUNG OS is the variant of hungarian matching that encourages over segments
        # HUNG is the standard hungarian matching algorithm
        assert self.match_mode in ['hung_os', 'hung']
        self.num_clusters = num_clusters        
        
    def forward(self, x):
                
        enc = self.pn_enc(x)
        pred = self.cluster_net(enc)        
        return pred
        
    def hung_match(self, pred, gt):

        gt_oh = F.one_hot(gt, self.num_clusters).float()        

        exp = bceloss(
            pred.view(-1, self.num_clusters, 1).repeat(1,1,self.num_clusters),
            gt_oh.view(-1, 1, self.num_clusters).repeat(1,self.num_clusters, 1)
        ).mean(dim=0)
        
        row_ind, col_ind = linear_sum_assignment(exp.cpu().numpy())

        re_ind = col_ind.argsort()
        
        return re_ind

    def hung_os_match(self, pred, gt):        
        try:
            hmatch = self.hung_match(pred, gt)        

        except Exception as e:
            utils.log_print(f"Failed hung match with {e}", self.args)
            return None, None
            
        m = {}
        
        uniqs = gt.unique()
        for r in uniqs:
            m[r.item()] = (gt == r).nonzero().flatten()

        # assigned map. keys are regions, values are the slots
        a = {}
        # unassigned list, tuples of (slot, unused gt region)
        u = []
        
        #for r, c in zip(hrow, hcol):
        for c, r in enumerate(hmatch):
            if c in m:
                a[c] = [r]
            else:
                u.append((r, c))

        ogt = gt.clone()

        pmx = pred.argmax(dim=1)

        # pred slot, gt slot
        for ps, gs in u:
            
            inds = (pmx == ps).nonzero().flatten()

            if inds.shape[0] < MIN_REG_POINTS:
                continue
            
            mode_region = ogt[inds].mode().values.item()

            purity = (ogt[inds] == mode_region).float().sum().item() / inds.shape[0] * 1.

            if purity <= MIN_REG_PURITY:
                continue
            
            reg_inds = m[mode_region]

            os = a[mode_region]
            
            change_inds = reg_inds[(pred[reg_inds, ps] > pred[reg_inds, os]).nonzero().flatten()]
            o_change_inds = reg_inds[(pred[reg_inds, ps] > pred[reg_inds, os]).nonzero().flatten()]            
            
            if change_inds.shape[0] < MIN_REG_POINTS:
                continue
            
            ogt[change_inds] = gs

            a[gs] = ps
            m[gs] = change_inds
            m[mode_region] = o_change_inds
            
        return hmatch, ogt        
        
    # pred is B x N X NC 
    # gt is B X N matrix --> assume max is NC

    def make_loss(self, pred, gt):                    
        return celoss(
            pred,
            gt
        )
    
    def make_match(self, bpred, bgt):
        if self.match_mode == 'hung':
            return (self.hung_match(bpred, bgt), bgt)
        
        elif self.match_mode == 'hung_os':
            return self.hung_os_match(bpred, bgt)
            
                    
    def loss(self, pred, gt, ex_weights = None):
                
        loss = 1e-8
        c = 1e-8

        accs = []
        
        for i in range(pred.shape[0]):
            bpred = pred[i]
            bgt = gt[i]

            with torch.no_grad():
                opt_gt_order, opt_gt = self.make_match(bpred, bgt)

                if opt_gt_order is None and opt_gt is None:
                    continue
                                
                accs.append(
                    (bpred[:, opt_gt_order].argmax(dim=1) == opt_gt).float().mean().item()
                )

            opt_loss = self.make_loss(bpred[:, opt_gt_order], opt_gt)

            if ex_weights is not None:
                opt_loss *= ex_weights[i]
            
            loss += opt_loss
            c += 1.

        return loss / c, torch.tensor(accs).mean().item()

                        
def load_model(args):
    net = SplitNet(
        args.sn_num_clusters,
        DP=args.dropout,
        use_bn = (args.use_bn == 'y'),
        match_mode = args.match_mode
    )            
    net.args = args
    return net
        

def fps(xyz, num_points):
    return pointnet2_utils.furthest_point_sample(
        xyz.contiguous(), num_points
    )
                        
