import sys, os, torch
import numpy as np
import utils
import train_utils
from utils import device
from copy import deepcopy
import random
from tqdm import tqdm
import math
import gc
import random
import time
from methods.merge_net.model import load_model
import region_ops as ro
import json

PRESAMP = None

def run_print_epoch(
    args, res, train_res, val_res, TRAIN_LOG_INFO, itn, ST
):
    json.dump(res, open(f"{args.outpath}/{args.exp_name}/res.json" ,'w'))
    
    t = time.time()
    
    utils.log_print(f"\nIter {itn} ({round(t - train_res['time'], 1)} / {round(t - ST, 1)}):", args)

    res['train_epochs'].append(itn)
            
    utils.log_print(
        f"Train results: ", args
    )
            
    utils.print_results(
        TRAIN_LOG_INFO,
        train_res,
        args,
    )

    utils.log_print(
        f"Val results: ", args
    )
            
    utils.print_results(
        TRAIN_LOG_INFO,
        val_res,
        args,
    )
            
    ep_result = {
        'train': train_res,
        'val': val_res
    }
            
    utils.make_ravg_plots(
        TRAIN_LOG_INFO,
        ep_result,
        res['train_plots'],
        res['train_epochs'],
        args,
        'train'
    )
            

def get_train_log_info(args):

    return [
        ('Loss', 'loss', 'batch_count'),
        ('Accuracy', 'corr', 'total'),
        ('Precision', 'pos_corr', 'prec_denom'),
        ('Recall', 'pos_corr', 'rec_denom'),
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
        self.iter_num = 0
        
        if not PRESAMP:
            self.data = data
            self.num_samples = len(data)
            self.eval_inds = list(range(self.num_samples))
            random.shuffle(self.eval_inds)
            self.eval_inds = self.eval_inds[:args.num_to_eval]            
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

                shape_regions = utils.clean_regions(shape_regions)
                
                self.data_shape_pts.append(shape_pts.cpu().numpy().astype('float16'))
                self.data_shape_norms.append(shape_normals.cpu().numpy().astype('float16'))
                self.data_shape_regs.append(shape_regions.cpu().numpy().astype('int16'))
                                                                           
        self.data_shape_pts = np.stack(self.data_shape_pts)
        self.data_shape_norms = np.stack(self.data_shape_norms)
        self.data_shape_regs = np.stack(self.data_shape_regs)                
        self.num_samples = self.data_shape_pts.shape[0]

        self.eval_inds = list(range(self.num_samples))
        random.shuffle(self.eval_inds)
        self.eval_inds = self.eval_inds[:args.num_to_eval]
        
    def make_synthetic_data(self):
        data_samps = []
        data_labels = []    

        if self.eval_only:
            
            inds = self.eval_inds

        else:
            inds = [
                random.randint(0, self.num_samples-1)
                for _ in range(self.args.num_samples_per_iter)
            ]
        

        for ind in inds:

            ind = random.randint(0, self.num_samples-1)
        
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
                    shape_regions = utils.clean_regions(shape_regions)
                    
                except Exception as e:
                    utils.log_print(f'Error in sampling {d.ind} -- {e}', self.args)
                    continue
            
            b_samps, b_labels = ro.merge_create_train_example(
                shape_pts,
                shape_normals,
                shape_regions,
                self.args,
            )
            
            for ex_samps, ex_labels in zip(b_samps, b_labels):
                data_samps.append(ex_samps.cpu().numpy().astype('float16'))
                data_labels.append(ex_labels)
                    
        data_samps = np.stack(data_samps)
        data_labels = np.array(data_labels)
        
        return data_samps, data_labels

        
    def __iter__(self):
        
        with torch.no_grad():
            data_samps, data_labels = self.make_synthetic_data()
            
        inds = torch.randperm(data_samps.shape[0])

        for start in range(0, inds.shape[0], self.batch_size):
        
            binds = inds[start:start+self.batch_size]

            if binds.shape[0] == 1:
                continue
            
            b_samps = torch.from_numpy(data_samps[binds]).float().to(device)
            b_labels = torch.from_numpy(data_labels[binds]).float().to(device)            
        
            assert len(b_samps.shape) == 3

            if not self.eval_only:
                with torch.no_grad():
                    scale = (torch.randn(b_samps.shape[0], 1, b_samps.shape[2], device=device) * self.scale_weight) + 1.
                    noise = torch.randn(b_samps.shape, device=device) * self.noise_weight
                
                    # Noise shouldn't affect normals
                    scale[:,:,3:] = 1.
                    noise[:,:,3:] = 0.
                
                    b_samps = (b_samps * scale) + noise

            self.iter_num += 1
            
            yield b_samps, b_labels
        
                               
def model_train_batch(batch, net, opt):
    samps, labels = batch
    preds = net(samps)

    loss, res = net.loss(preds, labels)
    
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    res['loss'] = loss.item()
        
    return res

def merge_net_load(args):    
    net = load_model(args)
    net.load_state_dict(
        torch.load(args.mn_path)
    )
    net.to(device)
    net.eval()
    return net
            
def train_merge_net(args, train_data, val_data):
    
    utils.init_model_run(args, 'merge_net')
    
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
    
    print(f"Num train samples  {train_loader.num_samples}")
    print(f"Num val samples {val_loader.num_samples}")
    
    net = load_model(args)
    net.to(device)
    
    opt = torch.optim.Adam(
        net.parameters(),
        lr = args.lr,
        eps = 1e-6
    )
    if args.sched == 'y':
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt,
            mode='min',
            factor = args.sched_factor,
            patience = args.sched_patience,
            min_lr = 1e-8,
            verbose=True
        )
    else:
        sched = None
        
    res = {
        'train_plots': {'train':{}, 'val':{}},
        'train_epochs': [],
    }

    save_model_weights = None

    best_itn = 0
    best = 0.

    last_print = 0

    TRAIN_LOG_INFO = get_train_log_info(args)

    cum_train_res = {'time': time.time()}
    
    T = time.time()
        
    while True:

        itn = train_loader.iter_num

        if itn > args.max_iters:
            break

        if itn - last_print > args.print_per:
            last_print = itn
            do_print = True
        else:
            do_print = False
            
        net.train()

        train_res = train_utils.model_train(
            train_loader,
            net,
            opt,
            model_train_batch
        )        
        
        for k,v in train_res.items():
            if k not in cum_train_res:
                cum_train_res[k] = 0.
            cum_train_res[k] += v
            
        if not do_print:
            continue

        if do_print:

            train_loss = cum_train_res['loss'] / cum_train_res['batch_count']

            if sched is not None:
                sched.step(train_loss)
            
            net.eval()
            with torch.no_grad():
                cum_val_res = train_utils.model_train(
                    val_loader,
                    net,
                    None,
                    model_train_batch
                )
            
                run_print_epoch(
                    args,
                    res,
                    cum_train_res,
                    cum_val_res,
                    TRAIN_LOG_INFO,
                    itn,
                    T
                )
                cum_train_res = {
                    'time': time.time()
                }

        acc_val = res['train_plots']['val']['Accuracy'][-1]
                    
        if acc_val > best + args.es_threshold:
            best = acc_val
            best_itn = itn
            
            save_model_weights = deepcopy(net.state_dict())

            torch.save(
                save_model_weights,
                f"{args.outpath}/{args.exp_name}/models/merge_net.pt"
            )
            
        if (itn - best_itn) > args.es_patience:
            utils.log_print("Stopping Early", args)
            break

                
class MERGE_NET:
    def __init__(self):
        self.name = 'merge_net'

        arg_list = [
            ('-o', '--outpath', 'methods/merge_net/model_output', str),
            ('-en', '--exp_name', None, str),
            ('-snp', '--shape_num_points', 100000, int),
            ('-mn_pnp', '--mn_part_num_points', 512, int),
            ('-nte', '--num_to_eval', 150, int),
            
            ('-rd', '--rd_seed', 42, int),
            ('-b', '--batch_size', 128, int),

            ('-prp', '--print_per', 1000, int),
            
            ('-nspi', '--num_samples_per_iter', 10, int),

            ('-mnet', '--max_num_example_tries', 3, int),
            ('-rwin', '--ravg_window', 10, int),
            
            ('-mi', '--max_iters', 10000000, int),
            ('-lr', '--lr', 0.0001, float),
            
            ('-esp', '--es_patience', 200000, int),
            ('-est', '--es_threshold', 0.0005, float),            
            
            ('-do', '--dropout', 0.0, float),
            ('-scalew', '--scale_weight', 0.0, float),
            ('-noisew', '--noise_weight', 0.0, float),

            ('-pres', '--presamp', 'y', str),
            ('-ubn', '--use_bn', 'y', str),

            ('-mn_ctx', '--mn_context', 0.1, float),
            
            # Scheduler logic

            ('-sch', '--sched', 'y', str),
            ('-scf', '--sched_factor', 0.25, float),
            ('-scp', '--sched_patience', 50, int),
            
            
        ]
        
        args = utils.getArgs(arg_list)        
        self.args = args
        global PRESAMP
        PRESAMP = (args.presamp == 'y')
        print(f"PRESAMP {PRESAMP}")
                
    def train(self, train_data, val_data, _ignore):

        args = self.args
        train_merge_net(args, train_data, val_data)






## EVAL LOGIC

def find_neighbors(points, regions, args, uvs):
    neighbors = []
    added = set()
    
    for r in uvs:
        added.add((r, r))
        q_pts = points[(regions == r).nonzero().flatten()[:1000]]

        d = torch.norm(q_pts.unsqueeze(1) - points.unsqueeze(0), dim=2)

        cd = (d <= args.nb_thresh).any(dim=0)

        nbs = regions[cd.nonzero().flatten()].cpu().unique().tolist()
        
        nbs = [(min(r,b), max(r,b)) for b in nbs]
        nbs = [k for k in nbs if k not in added]

        random.shuffle(nbs)
        
        for k in nbs[:args.max_nbs]:            
            neighbors.append(k)
            added.add(k)
                            
    return neighbors

def inf_merge(net, points, normals, _regs, args):
    regs = _regs.clone()

    uvs = regs.cpu().unique().tolist()
    uvs.sort()
    
    if len(uvs) <= 1:
        return regs, False
    
    neighbors = find_neighbors(points, regs, args, uvs)

    merge_examples = []
    merge_nbs = []
    
    for a,b in neighbors:

        ex_samps = ro.merge_format_example(
            (regs == a).nonzero().flatten(),
            (regs == b).nonzero().flatten(),
            points,
            normals,
            args
        )

        if ex_samps is None:
            continue
        
        merge_examples.append(ex_samps)        
        merge_nbs.append((a, b))
        
    if len(merge_examples) == 0:
        return regs, False

    merge_preds = []

    BS = args.batch_size
    merge_examples = torch.stack(merge_examples,dim=0)

    for i in range(0, merge_examples.shape[0], BS):
        pred = torch.sigmoid(net(merge_examples[i:i+BS].to(device))).cpu()
        merge_preds.append(pred)

    merge_preds = torch.cat(merge_preds, dim=0)
    
    ranked_merges = []

    for i, (a,b) in enumerate(merge_nbs):
        
        gain = merge_preds[i]

        if gain >= args.mn_merge_thresh:        
            ranked_merges.append((gain, (i, a,b)))

    ranked_merges.sort(reverse=True)

    merged = set()
    
    for pred_merge_lik, (i, a, b) in ranked_merges:
        if a in merged or b in merged:
            continue

        regs[(regs==b).nonzero().flatten()] = a
        merged.add(a)
        merged.add(b)
    
    return utils.clean_regions(regs), len(merged) > 0
