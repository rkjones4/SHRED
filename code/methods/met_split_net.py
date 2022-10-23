import sys, os, torch
import numpy as np
import utils
import train_utils
from utils import device
from copy import deepcopy
import random
from tqdm import tqdm
import utils
from methods.init_split import init_split
from methods.split_net.model import load_model

def get_log_info(args):
    return [
        ('Loss', 'loss', 'batch_count'),
        ('Accuracy', 'acc', 'batch_count'),
    ]
        
class Block:
    def __init__(self):
        pass


def make_train_blocks(data, args, training):
    
    samps, clusters, segments = init_split('regions', args, data)

    if samps is None:
        return []
            
    blocks = []
    
    for c in clusters.cpu().unique():
        in_pts = (clusters == c).nonzero().flatten()

        in_pts, _ = utils.shrink_inds(in_pts, args.sn_num_points_per_block)
                
        blk_samps = utils.normalize(samps[in_pts])
                
        b = Block()
                                    
        cpu_segments = segments[in_pts].cpu()        
        num_segments = cpu_segments.unique().shape[0]

        b.num_segments = num_segments
        b.samps = blk_samps.cpu()
        b.segments = cpu_segments                    
                                    
        blocks.append(b)

    return blocks

class Dataset:
    def __init__(
        self, data, args, eval_only
    ):

        self.scale_weight = args.scale_weight
        self.noise_weight = args.noise_weight
        
        self.eval_only = eval_only        
        
        batch_size, num_points, num_points_per_block = \
            args.batch_size, args.num_points, args.sn_num_points_per_block
        
        self.data_samps = []
        self.data_segments = []
        self.data_num_segs = []
        
        for d in tqdm(data):
            
            with torch.no_grad():
                blocks = make_train_blocks(
                    d,
                    args,
                    True
                )

                for b in blocks:

                    num_segs, samps, segments = \
                        b.num_segments, b.samps, b.segments

                    if num_segs < args.sn_min_clusters:
                        continue
                    
                    segments = utils.clean_regions(segments, args.sn_num_clusters)
                    
                    n_samps = samps.numpy().astype('float16')
                    n_segments = segments.numpy().astype('int16')

                    assert segments.cpu().unique().shape[0] <= args.sn_num_clusters
                    self.data_num_segs.append(min(num_segs, args.sn_num_clusters))
                    self.data_samps.append(n_samps)
                    self.data_segments.append(n_segments)
                            
        self.data_samps = np.stack(self.data_samps)
        self.data_segments = np.stack(self.data_segments)
        self.data_num_segs = torch.tensor(self.data_num_segs)

        self.batch_size = batch_size

    def __iter__(self):

        inds = torch.randperm(self.data_samps.shape[0])
            
        for start in range(0, inds.shape[0], self.batch_size):
            
            binds = inds[start:start+self.batch_size]
            
            b_samps = torch.from_numpy(self.data_samps[binds]).float().to(device)
            b_segments = torch.from_numpy(self.data_segments[binds]).long().to(device)            
            
            if len(b_samps.shape) == 2:
                b_samps = b_samps.unsqueeze(0)
                b_segments = b_segments.unsqueeze(0)

            if not self.eval_only:
                with torch.no_grad():
                    scale = (torch.randn(b_samps.shape[0], 1, b_samps.shape[2], device=device) * self.scale_weight) + 1.
                    noise = torch.randn(b_samps.shape, device=device) * self.noise_weight

                    # Noise shouldn't affect normals
                    scale[:,:,3:] = 1.
                    noise[:,:,3:] = 0.
                    
                    b_samps = (b_samps * scale) + noise    
                    
            yield b_samps, b_segments
                               
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

def train_split_net(args, train_data, val_data):
    
    utils.init_model_run(args, 'split_net')
                                
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

    print(f"Num train examples {train_loader.data_samps.shape[0]}")
    print(f"Num val examples {val_loader.data_samps.shape[0]}")
    
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
                f"{args.outpath}/{args.exp_name}/models/split_net.pt"
            )
            
        if (e - best_ep) > args.es_patience:
            utils.log_print("Stopping Early", args)
            break

def split_net_load(args):
    net = load_model(args)
    net.load_state_dict(torch.load(
        f'{args.sn_path}'
    ))
    net.eval()
    net.to(device)

    return net


def local_split(net, args, q_samps, k_samps, i_samps):
    
    preds = net(i_samps.unsqueeze(0))
    pred_regs = utils.clean_regions(preds[0].argmax(dim=1))    
    
    closest = (q_samps[:,:3].unsqueeze(1) - k_samps[:,:3].unsqueeze(0)).norm(dim=2).argmin(dim=1)
    
    return pred_regs[closest]    


def model_split(net, args, samps, regions):
    
    offset = 0
    new_regions = torch.zeros(regions.shape, device=regions.device).long() - 1
    
    for s in regions.cpu().unique().tolist():

        q_inds = (regions == s).nonzero().flatten()
        k_inds, _ = utils.shrink_inds(q_inds, args.sn_num_points_per_block)

        q_samps = samps[q_inds]
        k_samps = samps[k_inds]
        i_samps = utils.normalize(k_samps)        

        local_pred = local_split(net, args, q_samps, k_samps, i_samps)
        
        new_regions[q_inds] = local_pred + offset
        
        offset += local_pred.cpu().unique().shape[0]

    assert (new_regions >= 0).all()

    return new_regions

        
class SPLIT_NET:
    def __init__(self):
        self.net = None
        self.name = 'split_net'

        arg_list = [
            ('-o', '--outpath', 'methods/split_net/model_output', str),
            ('-en', '--exp_name', None, str),
            ('-np', '--num_points', 100000, int),
                       
            ('-ism', '--init_split_mode', 'fps', str),
            ('-inb', '--init_num_blocks', 64, int),            

            ('-sn_nppb', '--sn_num_points_per_block', 512, int),
            ('-sn_minc', '--sn_min_clusters', 1, int),
            ('-sn_nc', '--sn_num_clusters', 10, int),

            ('-mm', '--match_mode', 'hung_os', str), 
            ('-rd', '--rd_seed', 42, int),
            ('-b', '--batch_size', 24, int),
            ('-e', '--epochs', 1000, int),
            ('-lr', '--lr', 0.001, float),
            ('-prp', '--print_per', 1, int),
            ('-esp', '--es_patience', 20, int),
            ('-est', '--es_threshold', 0.001, float),            
            ('-ubn', '--use_bn', 'y', str),
            
            ('-do', '--dropout', 0.1, float),
            ('-scalew', '--scale_weight', 0.0, float),
            ('-noisew', '--noise_weight', 0.0, float),                     
        ]

        args = utils.getArgs(arg_list)
        
        self.args = args
                               
    def train(self, train_data, val_data, _ignore):
        train_split_net(self.args, train_data, val_data)

        
