import os
import json
import utils
import numpy as np
from tqdm import tqdm
import torch
import sys

RESULT_DIR = 'results'
SAVE_DIR = 'save_results'
os.system(f'mkdir {RESULT_DIR} > /dev/null 2>&1')
os.system(f'mkdir {SAVE_DIR} > /dev/null 2>&1')

MAX = None

class ShapeData:
    def __init__(self, ind, cat):
        data = json.load(open(f'../data/{cat}/{ind}/data.json'))
        regions = [utils.loadAndCleanObj(p) for p in data['regions']]
        dummy_labels = np.array([
            -1 for label in data['sem_labels']
        ])
        part_ids = np.array(data['part_ids'])

        self.ind = ind
        self.cat = cat
        self.DEBUG_regions = regions
        self.DEBUG_part_ids = part_ids

        # Methods get access to mesh, points, normals + ind (caching)
        # EVAL is for metrics
        # DEBUG is for debugging
        
        self.mesh, self.points, self.normals, self.EVAL_part_ids, _ = \
            utils.eval_sample(
                regions, part_ids, dummy_labels
            )        

def load_data(cat):
    inds = json.load(open(f'data_splits/{cat}/split.json'))['test']

    if MAX is not None:
        inds = inds[:MAX]

    
    data = [
        ShapeData(ind, cat) for ind in tqdm(inds)
    ]

    for i,d in enumerate(data):
        d.count_ind = i

    return data
    

def make_preds(method, data):
    preds = []

    for d in tqdm(data):
        preds.append(method.make_pred(d))
        
    return preds

def calc_purity(pred, gt, la):

    purs = []

    for i in gt.unique():
        inds = (gt == i).nonzero().flatten()
        la_regs = la[inds]

        p = (la_regs == i).float().mean().item()

        purs.append(p)
        
    return torch.tensor(purs).mean().item()
        
def getBestAssign(R, G):
    A = torch.zeros(R.shape[0]).long() - 1

    for i in R.unique().tolist():
        inds = (R == i).nonzero().flatten()
        A[inds] = G[inds].mode().values.item()
        
    return A

def calc_inst_seg_metrics(all_preds, all_gt):

    all_gt_best_ious = []

    for pred, gt in zip(all_preds, all_gt):

        p_exp = torch.nn.functional.one_hot(pred)
        gt_best_ious = []
        for i in gt.unique():
            gt_exp = (gt == i).float()
            comb = gt_exp.view(-1, 1) + p_exp
            ious = (comb == 2).float().sum(dim=0) / (comb > 0).float().sum(dim=0)
            gt_best_ious.append(ious.max().item())

        all_gt_best_ious.append(torch.tensor(gt_best_ious))

    all_gt_best_ious = torch.cat(all_gt_best_ious, dim=0)

    aiou = all_gt_best_ious.mean().item()

    return {        
        'inst_aiou': aiou,
    }        


def calc_metrics(preds, data, category, res_name):

    results = {
        'num_regions': [],
        'pur': [],
    }

    raw_pt_inst_preds = []
    pt_inst_labels = []
    
    for P, D in zip(preds, data):
        
        pt_inst_labels.append(D.EVAL_part_ids)
        raw_pt_inst_preds.append(P)
        
        results['num_regions'].append(P.unique().shape[0] * 1.)
        
        A = getBestAssign(P, D.EVAL_part_ids)

        pur = calc_purity(P, D.EVAL_part_ids, A)

        results['pur'].append(pur)
        
    results = {k:torch.tensor(v).float().mean().item() for k,v in results.items()}

    inst_res = calc_inst_seg_metrics(raw_pt_inst_preds, pt_inst_labels)
    
    results.update({f'raw_{k}':v for k,v in inst_res.items()})

    save_preds = torch.stack(raw_pt_inst_preds, dim=0).cpu().numpy().astype('int16')
    np.save(f'{SAVE_DIR}/{res_name}/preds_{category}.npy', save_preds)    
    
    return results


def eval_method_per_cat(category, method, res_name):

    print(category)
    
    # data is list of (input mesh, sampled points sampled part_ids, sampled sem_labels) 
    data = load_data(category)

    # preds is list of (predicted part_ids)
    preds = make_preds(method, data)

    metrics = calc_metrics(preds, data, category, res_name)

    return metrics

def save_data(method, save_name, num, IN_DOMAIN, OUT_DOMAIN):
    if num is not None:
        global MAX
        MAX = num

    os.system(f'mkdir {save_name} > /dev/null 2>&1')
        
    for cat in IN_DOMAIN + OUT_DOMAIN:
        print(cat)
        data  = load_data(cat)
        method.save_data(data, f'{save_name}/{cat}')
        
def eval_method(method, res_name, num, IN_DOMAIN, OUT_DOMAIN):

    os.system(f'mkdir {SAVE_DIR}/{res_name} > /dev/null 2>&1')
    
    if num is not None:
        global MAX
        MAX = num
    
    results = {}

    in_res= {}
    out_res = {}
    
    for cat in IN_DOMAIN:
        m = eval_method_per_cat(cat, method, res_name)

        results[cat] = m
        
        for k,v in m.items():
            if k not in in_res:
                in_res[k] = []
                
            in_res[k].append(v)

    for cat in OUT_DOMAIN:
        m = eval_method_per_cat(cat, method, res_name)

        results[cat] = m
        
        for k,v in m.items():
            if k not in out_res:
                out_res[k] = []
                
            out_res[k].append(v)

    in_res = {k: torch.tensor(v).float().mean().item() for k,v in in_res.items()}
    out_res = {k: torch.tensor(v).float().mean().item() for k,v in out_res.items()}

    results['in_domain'] = in_res
    results['out_domain'] = out_res

    print(results)
    
    for k,v in results.items():
        r = ' '.join([f'{name}:{round(value,3)}' for name,value in v.items()])
        print(f'{k} --> {r}')

    json.dump(results, open(f'{RESULT_DIR}/{res_name}.json', 'w'))
