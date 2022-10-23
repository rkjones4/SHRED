import sys, os, torch
import numpy as np
import utils
import train_utils
from utils import device
from copy import deepcopy
import random
from tqdm import tqdm
import math
from methods.fix_net.model import load_model
import gc
import region_ops as ro

PRESAMP = None

def get_log_info(args):
    return [
        ('Loss', 'loss', 'batch_count'),
        ('Accuracy', 'acc', 'batch_count'),
    ]
                                                        
class Dataset:
    def __init__(
        self, data, args, eval_only
    ):

        self.args = args
        
        self.scale_weight = args.scale_weight
        self.noise_weight = args.noise_weight
        
        self.eval_only = eval_only            
        
        self.batch_size = args.batch_size
                
        if not PRESAMP:
            self.data = data
            self.num_samples = len(data)
            return
    
        self.data_shape_pts = []
        self.data_shape_norms = []
        self.data_shape_regs = []
    
        for d in tqdm(data):                        
            with torch.no_grad():                
                try:
                    _, shape_pts, shape_normals, shape_regions = utils.train_sample(
                        d.regions,
                        d.part_ids,
                        num_points=args.shape_num_points        
                    )
                    
                except Exception as e:
                    utils.log_print(f'Error in sampling {d.ind} -- {e}', self.args)
                    continue

                self.data_shape_pts.append(shape_pts.cpu().numpy().astype('float16'))
                self.data_shape_norms.append(shape_normals.cpu().numpy().astype('float16'))
                self.data_shape_regs.append(shape_regions.cpu().numpy().astype('int16'))
                                                                           
        self.data_shape_pts = np.stack(self.data_shape_pts)
        self.data_shape_norms = np.stack(self.data_shape_norms)
        self.data_shape_regs = np.stack(self.data_shape_regs)                
        self.num_samples = self.data_shape_pts.shape[0]

    def make_synthetic_data(self):
        data_samps = []
        data_labels = []
        print("Making Synthetic Data")
        self.args.cnt = 0
        
        for ind in tqdm(list(range(self.num_samples))):

            if PRESAMP:
                shape_pts = torch.from_numpy(self.data_shape_pts[ind]).float().to(device)
                shape_normals = torch.from_numpy(self.data_shape_norms[ind]).float().to(device)
                shape_regions = torch.from_numpy(self.data_shape_regs[ind]).long().to(device)

            else:
                d = self.data[ind]
                try:
                    _, shape_pts, shape_normals, shape_regions = utils.train_sample(
                        d.regions,
                        d.part_ids,
                        num_points=self.args.shape_num_points        
                    )
                    
                except Exception as e:
                    utils.log_print(f'Error in sampling {d.ind} -- {e}', self.args)
                    continue
                
                
            for s in shape_regions.cpu().unique().tolist():
                                                                                    
                ex_samps, ex_labels = ro.fix_create_train_example(
                    shape_pts,
                    shape_normals,
                    shape_regions,
                    s,
                    self.args,
                    self.args.fn_cte_mode
                )
                    
                if ex_samps is None or ex_labels is None:      
                    continue

                data_samps.append(ex_samps.cpu().numpy().astype('float16'))
                data_labels.append(ex_labels.cpu().numpy().astype('float16'))                

        data_samps = np.stack(data_samps)
        data_labels = np.stack(data_labels)
        
        return data_samps, data_labels
        
    def __iter__(self):
        
        with torch.no_grad():
            data_samps, data_labels = self.make_synthetic_data()

        inds = torch.randperm(data_samps.shape[0])

        for start in range(0, inds.shape[0], self.batch_size):
        
            binds = inds[start:start+self.batch_size]
        
            b_samps = torch.from_numpy(data_samps[binds]).float().to(device)
            b_labels = torch.from_numpy(data_labels[binds]).float().to(device)            
        
            if len(b_samps.shape) == 2:
                b_samps = b_samps.unsqueeze(0)
                b_labels = b_labels.unsqueeze(0)

            if not self.eval_only:
                with torch.no_grad():
                    scale = (torch.randn(b_samps.shape[0], 1, b_samps.shape[2], device=device) * self.scale_weight) + 1.
                    noise = torch.randn(b_samps.shape, device=device) * self.noise_weight
                
                    # Noise shouldn't affect normals
                    scale[:,:,3:] = 1.
                    noise[:,:,3:] = 0.
                
                    b_samps = (b_samps * scale) + noise    
                
            yield b_samps, b_labels
        
                   
            

def model_train_batch(batch, net, opt):
    samps, segments = batch
    
    br = {}

    preds = net(samps)
    
    loss, acc = net.loss(preds, segments)
    
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    br['loss'] = loss.item()
    br['acc'] = acc
    
    return br

def train_fix_net(args, train_data, val_data):
    
    utils.init_model_run(args, 'fix_net')
                                
    train_loader = Dataset(
        train_data,
        args,
        False
    )

    val_loader = Dataset(
        val_data,
        args,
        True
    )        

    if PRESAMP:
    
        del train_data
        del val_data
        gc.collect()
    
    print(f"Num train voxels {train_loader.num_samples}")
    print(f"Num val voxels {val_loader.num_samples}")
    
    net = load_model(args)
    net.to(device)
    
    opt = torch.optim.Adam(
        net.parameters(),
        lr = args.lr,
        eps = 1e-6
    )
        
    res = {
        'train_plots': {'train':{}, 'val':{}},
        'train_epochs': [],
    }

    save_model_weights = None

    best_ep = 0.
    best_loss = 1e8
    
    train_fn = model_train_batch

    LOG_INFO = get_log_info(args)
        
    for e in range(args.epochs):
        
        train_utils.run_train_epoch(
            args,
            res,
            net,
            opt,
            train_loader,
            val_loader,
            LOG_INFO,
            e,
            train_fn
        )

        val_loss = res['train_plots']['val']['Loss'][-1]
                    
        if val_loss + args.es_threshold < best_loss:
            best_loss = val_loss
            best_ep = e            
            
            save_model_weights = deepcopy(net.state_dict())

            torch.save(
                save_model_weights,
                f"{args.outpath}/{args.exp_name}/models/fix_net.pt"
            )
            
        if (e - best_ep) > args.es_patience:
            utils.log_print("Stopping Early", args)
            break


def fix_net_load(args):
    net = load_model(args)
    net.load_state_dict(torch.load(
        f'{args.fn_path}'
    ))
    net.eval()
    net.to(device)

    return net

COUNT = [0]

def local_fix(net, args, samps, input_labels):

    ex_samps, query_inds, key_pts = ro.create_fix_eval_example(
        samps, input_labels, args.fn_context, args.fn_part_num_points, args.fn_cte_mode
    )        
    
    pred = torch.sigmoid(net(ex_samps.unsqueeze(0))[0])

    closest = torch.zeros(query_inds.shape[0], dtype=torch.long, device=pred.device)

    BS = 10000
    
    for i in range(0, query_inds.shape[0], BS):
        closest[i:i+BS] = (samps[query_inds[i:i+BS]][:,:3].unsqueeze(1) - key_pts.unsqueeze(0)).norm(dim=2).argmin(dim=1)
    
    c_pred = pred[closest]        
    
    local_pred = torch.ones(input_labels.shape, device=c_pred.device) * -1.
    local_pred[query_inds] = c_pred        

    return local_pred


def model_fix(net, args, samps, regions):

    uniqs = regions.cpu().unique().tolist()
    uniqs.sort()

    fix_preds = torch.zeros(regions.shape[0], len(uniqs),  device=regions.device).float()
    
    for i,s in enumerate(uniqs):

        assert i == s
        
        pred = local_fix(net, args, samps, (regions == s))        
        
        fix_preds[:, i] = pred
                
    new_regions = fix_preds.argmax(dim=1)    
    
    return utils.clean_regions(new_regions)
        
        
class FIX_NET:
    def __init__(self):
        self.name = 'fix_net'

        arg_list = [
            ('-o', '--outpath', 'methods/fix_net/model_output', str),
            ('-en', '--exp_name', None, str),
            ('-snp', '--shape_num_points', 100000, int),            
            ('-mnet', '--max_num_example_tries', 4, int),
            
            ('-rd', '--rd_seed', 42, int),
            ('-b', '--batch_size', 64, int),
            ('-e', '--epochs', 1000, int),
            ('-lr', '--lr', 0.0001, float),
            ('-prp', '--print_per', 1, int),
            ('-esp', '--es_patience', 20, int),
            ('-est', '--es_threshold', 0.001, float),            
            
            ('-do', '--dropout', 0.1, float),
            ('-scalew', '--scale_weight', 0.025, float),
            ('-noisew', '--noise_weight', 0.005, float),

            ('-pres', '--presamp', 'y', str),

            ## Example creation params
            ('-fn_pnp', '--fn_part_num_points', 2048, int),
            ('-fn_ctx', '--fn_context', 0.1, float),            
            ('-fn_cte_mode', '--fn_cte_mode', 'def_bal', str)
        ]        
        args = utils.getArgs(arg_list)        
        self.args = args
        global PRESAMP
        PRESAMP = (args.presamp == 'y')
        print(f"PRESAMP {PRESAMP}")

        args.cte_flip_perc = 0.3
        args.cte_change_thresh = 0.5
        args.cte_add_chance = 0.75
        args.cte_rem_chance = 0.75
        args.cte_min_change_perc = 0.5
        args.cte_min_reg_pts = 10
        args.cte_add_min = 0.05
        args.cte_add_range = 0.2
        args.cte_rem_min = 0.1
        args.cte_rem_range = 0.4
        args.cte_num_extra_perc = 0.1
            
        args.cte_part_num_points = args.fn_part_num_points
        args.cte_context = args.fn_context
        
    def train(self, train_data, val_data, _ignore):

        args = self.args
        train_fix_net(args, train_data, val_data)

        
