import os
import json
from utils import IN_DOMAIN
import utils
import numpy as np
from tqdm import tqdm
import torch
import sys

MAX = None

class TrainShapeData:
    def __init__(self, ind, cat):
        data = json.load(open(f'../data/{cat}/{ind}/data.json'))
        regions = [utils.loadAndCleanObj(p) for p in data['regions']]        
        part_ids = np.array(data['part_ids'])
        self.ind = ind
        self.regions = regions
        self.part_ids = part_ids                
        self.json_data = data
        
def load_data(cat, split, split_path):
    inds = json.load(open(f'{split_path}/{cat}/split.json'))[split]

    if MAX is not None:
        inds = inds[:MAX]
    
    data = [
        TrainShapeData(ind, cat) for ind in tqdm(inds)
    ]

    return data
    

def train_method(method, save_name, split_path):

    train_data = []
    val_data = []    

    print("Loading Data")
    
    for cat in args.cats:
        print(cat)
        train_data += load_data(cat, 'train', split_path)
        val_data += load_data(cat, 'val', split_path)
        
    method.train(train_data, val_data, save_name)

def main(args):

    global MAX
    
    if args.max is not None:
        MAX = args.max

    # BASELINE NETWORKS
    if args.method == 'pn_seg':
        from methods.baselines.met_pn_seg import PN_SEG
        M = PN_SEG()
        
    elif args.method == 'wopl':
        from methods.baselines.met_wopl import WOPL
        M = WOPL()

    # SHRED NETWORKS
    elif args.method == 'split_net':
        from methods.met_split_net import SPLIT_NET
        M = SPLIT_NET()
        
    elif args.method == 'fix_net':
        from methods.met_fix_net import FIX_NET
        M = FIX_NET()

    elif args.method == 'merge_net':
        from methods.met_merge_net import MERGE_NET
        M = MERGE_NET()    
        
    if args.cats is None:
        args.cats = IN_DOMAIN
    else:
        args.cats = args.cats.split(',')
        
    train_method(M, args.name, args.split_path)


if __name__ == '__main__':

    arg_list = [
        ('-mt', '--method', None,  str),        
        ('-en', '--name', None, str),
        ('-mx', '--max', None, int),
        ('-cats', '--cats', None, str),
        ('-sp', '--split_path', 'data_splits', str)
    ]        

    args = utils.getArgs(arg_list)    
    
    main(args)
    

    


    
