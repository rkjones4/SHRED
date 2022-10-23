import torch, math, random
import utils
from utils import device
from copy import deepcopy
from methods.split_net.model import fps


# UTILS FOR REGION CREATION

def get_center_and_rad(in_pts):
    cen = (in_pts.max(dim=0).values + in_pts.min(dim=0).values) / 2
    rad = torch.norm(in_pts-cen, dim=1).max()
    exp_rad = (in_pts - cen).abs().max(dim=0).values    
    
    return cen, rad, exp_rad

def get_reg_info(pts, regs, s_id):
    in_reg = (regs == s_id)
    
    in_pts = pts[in_reg.nonzero().flatten()]
    
    cen, rad, exp_rad = get_center_and_rad(in_pts)
            
    return in_reg, cen, rad, exp_rad            


## START MERGE LOGIC


FPS_OPTS = [16, 32, 64, 128]
MK = 10
ADJ_THRESH = 0.025

def sim_fps_split(pts):
    S = int(random.random() * (pts.shape[0]/2.))

    s_pts = pts[S:S+1000]

    NC = random.choice(FPS_OPTS)
    
    c_inds = fps(s_pts.unsqueeze(0), NC)[0].long()

    centers = s_pts[c_inds]

    dist = (pts.view(-1,1,3) - centers.view(1,-1,3)).norm(dim=2)
    fps_clusters = dist.argmin(dim=1)
        

    return fps_clusters

SR_dist = torch.tensor([(.5**x) for x in range(1,MK+1)])

def split_part(inds, pts):

    NS = torch.distributions.Categorical(SR_dist).sample().item() + 1
    
    if NS == 1:
        return [inds]
    
    cen, _, exp = get_center_and_rad(pts)

    q_pts = cen.view(1,3).repeat(NS,1) + \
        (torch.randn(NS, 3, device=pts.device) * exp.view(1,3))

    D = torch.norm(pts.unsqueeze(1) - q_pts.unsqueeze(0), dim = 2)
    
    QC = D.argmin(dim=1)
    
    splits = []

    for q in QC.cpu().unique():
        splits.append(inds[(QC == q).nonzero().flatten()])    
    
    return splits
    

def sim_part_split(pts, gt_regs):
    regs = torch.zeros(gt_regs.shape, device=gt_regs.device, dtype=torch.long)

    for r in gt_regs.cpu().unique().tolist():
        inds = (gt_regs == r).nonzero().flatten()
        r_pts = pts[inds]        
        
        splits = split_part(inds, r_pts)

        for split_inds in splits:
            rnd = random.random()

            rv = random.randint(0, MK)
            
            if rnd < 0.33:
                # keep at 0
                pass
            elif rnd < 0.66:
                regs[split_inds] = rv

            else:
                regs[split_inds] = rv * r
        
    return regs

def find_nbs(pts, regs, gt_regs):

    reg_inds = []
    reg_gt_val = []
    
    # N x 128 x 3    
    RS = []
    
    for r in regs.cpu().unique().tolist():
        inds = (regs == r).nonzero().flatten()
        reg_inds.append(inds)

        _inds, _ = utils.shrink_inds(inds, 128)
        
        RS.append(pts[_inds])

        reg_gt_val.append(gt_regs[inds].mode().values.item())

    RS = torch.stack(RS,dim=0)

    nb_mat = torch.zeros((RS.shape[0], RS.shape[0]), dtype=torch.bool, device = RS.device)

    BS = 100
    
    for i in range(0,RS.shape[0], BS):
        for j in range(0, RS.shape[0], BS):
            D = torch.norm(RS[i:i+BS].unsqueeze(1).unsqueeze(-2) - RS[j:j+BS].unsqueeze(0).unsqueeze(-3), dim =-1)
            nb_mat[i:i+BS,j:j+BS] = D.min(dim=3).values.min(dim=2).values < ADJ_THRESH
        
    nb_mat[torch.arange(RS.shape[0]), torch.arange(RS.shape[0])] = False
    
    return nb_mat, reg_inds, reg_gt_val
    
    
    
def sim_merge_round(pts, gt_regs):

    # [0, #FPS]
    fps_regs = sim_fps_split(pts)

    # if 0-MK, assigned to fps region, but can group
    # if > MK, then unique to part, but still grouped
    # [0, #MK+1], unique to itself
    
    part_regs = sim_part_split(pts, gt_regs)

    regs = fps_regs + (part_regs * fps_regs.max().item())
    
    # nb_mat is K x K matrix, True if negihbors
    # reg_inds is list of K elements to current inds that demarcate that region
    # reg_gt_val is lits of K elements to best gt region for each region
        
    nb_mat, reg_inds, reg_gt_val = find_nbs(pts, regs, gt_regs)
    
    examples = []

    novel = torch.ones(nb_mat.shape[0], dtype=torch.bool)
    
    while nb_mat.any():

        a,b = random.choice(nb_mat.nonzero())

        assert reg_gt_val[a] is not None and\
            reg_gt_val[b] is not None and\
            reg_inds[a] is not None and\
            reg_inds[b] is not None

        gt_do_merge = reg_gt_val[a] == reg_gt_val[b]
        examples.append((reg_inds[a], reg_inds[b], gt_do_merge))            
            
        do_merge_thresh = 0.75 if reg_gt_val[a] == reg_gt_val[b] else 0.25

        do_merge = random.random() <= do_merge_thresh
            
        nb_mat[a, b] = False
        nb_mat[b, a] = False

        if do_merge:
            nb_mat[a,:] = nb_mat[a,:] | nb_mat[b,:]
            nb_mat[:,a] = nb_mat[:,a] | nb_mat[:,b]
            
            nb_mat[b,:] = False
            nb_mat[:,b] = False

            reg_inds[a] = torch.cat((reg_inds[a], reg_inds[b]))
            reg_inds[b] = None

            reg_gt_val[a] = gt_regs[reg_inds[a]].mode().values.item()
            reg_gt_val[b] = None

            # added last time
            novel[a] = False
            novel[b] = False
                        
    return examples

        

def merge_format_example(ainds, binds, s_pts, s_norms, args):

    if ainds.shape[0] == 0 or binds.shape[0] == 0:
        return None

    input_labels = torch.zeros(s_pts.shape[0], device=s_pts.device).float()
    input_labels[ainds] = 1.0
    input_labels[binds] = 2.0

    input_inds = input_labels.nonzero().flatten()

    context = args.mn_context
    N = args.mn_part_num_points
    
    n_cen, n_rad, _ = get_center_and_rad(s_pts[input_inds])

    n_brad = n_rad.item() + context

    m_pts = s_pts - n_cen

    in_outer_rad = (m_pts.norm(dim=1) <= n_brad)

    big_inp_neg_inds = ((input_labels == 0.) & in_outer_rad).nonzero().flatten()    

    if big_inp_neg_inds.shape[0] < 1:
        return None
        
    inp_a_inds = (input_labels == 1.0).nonzero().flatten()[:N]
    inp_b_inds = (input_labels == 2.0).nonzero().flatten()[:N]

    inp_neg_inds, _ = utils.shrink_inds(
        big_inp_neg_inds,
        (4*N) - inp_a_inds.shape[0] - inp_b_inds.shape[0]
    )

    comb_inds = torch.cat((inp_a_inds, inp_b_inds, inp_neg_inds), dim =0)
    
    inp_labels = torch.zeros(4*N, 2, device=comb_inds.device)
    inp_labels[:inp_a_inds.shape[0], 0] = 1.0
    inp_labels[inp_a_inds.shape[0]:inp_a_inds.shape[0]+inp_b_inds.shape[0], 1] = 1.0
    
    inp_pts = s_pts[comb_inds].clone()
    inp_norms = s_norms[comb_inds].clone()

    norm_pts = utils.normalize(inp_pts)

    ex_samps = torch.cat((norm_pts, inp_norms, inp_labels), dim =1)    
    
    return ex_samps.cpu()
    

def merge_create_train_example(s_pts, s_norms, s_regs, args):

    examples = sim_merge_round(
        s_pts,
        s_regs,
    )

    b_samps, b_labels = [], []

    tc = 0
    fc = 0
    
    for ainds, binds, label in examples:
        ex_samps = merge_format_example(ainds, binds, s_pts, s_norms, args)

        if ex_samps is None:
            continue        
            
        b_samps.append(ex_samps)
        b_labels.append(float(label))

    return b_samps, b_labels
    


## END MERGE LOGIC

## START FIX LOGIC

def bal_fix_create_eval_example(samps, input_labels, context, N):
        
    n_cen, n_rad, _ = get_center_and_rad(samps[input_labels.nonzero().flatten(),:3])

    n_brad = n_rad.item() + context

    m_pts = samps[:,:3] - n_cen

    in_outer_rad = (m_pts.norm(dim=1) <= n_brad)
    
    big_inp_pos_inds = input_labels.nonzero().flatten()
    big_inp_neg_inds = ((~input_labels) & in_outer_rad).nonzero().flatten()

    inp_pos_inds, _ = utils.shrink_inds(big_inp_pos_inds, N)
    inp_neg_inds, _ = utils.shrink_inds(big_inp_neg_inds, N)

    inp_neg_vals = torch.zeros(N,device=samps.device)
    inp_pos_vals = torch.ones(N,device=samps.device)
        
    comb_inds = torch.cat((inp_pos_inds, inp_neg_inds), dim=0)

    inp_labels = torch.cat((inp_pos_vals, inp_neg_vals), dim = 0)

    inp_pts = samps[comb_inds,:3]
    inp_norms = samps[comb_inds,3:]

    norm_pts = utils.normalize(inp_pts)

    ex_samps = torch.cat((norm_pts, inp_norms, inp_labels.view(-1,1)), dim =1)

    query_inds = in_outer_rad.nonzero().flatten()

    key_pts = inp_pts
    
    return ex_samps, query_inds, key_pts


def create_fix_eval_example(samps, input_labels, context, N, mode):
    assert 'bal' in mode
    
    return bal_fix_create_eval_example(samps, input_labels, context, N)


def fix_create_train_example(s_pts, s_norms, s_regs, s_id, args, mode):

    s_info = get_reg_info(s_pts, s_regs, s_id)

    for _ in range(args.max_num_example_tries):
        
        samp_input_labels = fix_sample_region_change(s_pts, s_info, args, mode)

        if samp_input_labels is None:
            continue
        
        ex_samps, ex_labels = fix_sample_example(samp_input_labels, s_pts, s_norms, s_info, args, mode)

        if ex_samps is None:
            continue

        return ex_samps, ex_labels
        
    return None, None


def fix_sample_region_change(s_pts, s_info, args, mode):
            
    ADD_CHANCE = args.cte_add_chance
    REM_CHANCE = args.cte_rem_chance
    
    ADD_MIN = args.cte_add_min
    ADD_RANGE = args.cte_add_range

    REM_MIN = args.cte_rem_min
    REM_RANGE = args.cte_rem_range
        
    in_reg, cen, rad, exp_rad = s_info        
    num_in_reg = in_reg.float().sum().item()
    
    input_labels = in_reg.clone()

    num_in_reg = input_labels.float().sum().item()    
    MIN_PTS = num_in_reg * args.cte_min_change_perc
    
    if random.random() < ADD_CHANCE:

        n_cen = cen + (torch.randn(3, device=device) * exp_rad / 2.)
        n_ext = (exp_rad / 2.) * torch.clamp((torch.randn(3, device=device) + 1.0), 0.01, 2.0)
                        
        v = -1 * ((s_pts - n_cen) / (n_ext.abs() + 0.01)).norm(dim=1)                

        v[input_labels.nonzero().flatten()] = -1e8
        
        nx_perc = (random.random() * ADD_RANGE) + ADD_MIN
        
        num_to_flip = min(round(nx_perc * num_in_reg), v.shape[0])
        inds_to_flip = torch.topk(v, num_to_flip).indices
        
        input_labels[inds_to_flip] = True
        
    if random.random() < REM_CHANCE:        

        n_cen = cen + torch.randn(3, device=device) * exp_rad
        n_ext = exp_rad * torch.clamp((torch.randn(3, device=device) + 1.0), 0.01, 2.0)
                        
        v = -1 * ((s_pts - n_cen) / (n_ext.abs() + 0.01)).norm(dim=1)                

        v[(~input_labels).nonzero().flatten()] = -1e8
        
        nx_perc = (random.random() * REM_RANGE) + REM_MIN

        num_to_flip = min(round(nx_perc * num_in_reg), v.shape[0])
        inds_to_flip = torch.topk(v, num_to_flip).indices
        
        input_labels[inds_to_flip] = False    


    if input_labels.float().sum().item() < MIN_PTS:
        return None
    
    return input_labels


def fix_sample_example(input_labels, s_pts, s_norms, s_info, args, mode):
    
    assert 'bal' in mode
    
    return fix_bal_sample_example(input_labels, s_pts, s_norms, s_info, args)

    
def fix_bal_sample_example(input_labels, s_pts, s_norms, s_info, args):

    context = args.cte_context
    flip_perc = random.random() * args.cte_flip_perc
    CHANGE_THRESH = args.cte_change_thresh    
    
    in_reg, cen, rad, exp_rad = s_info
    
    NUM_EXTRA_PERC = args.cte_num_extra_perc
        
    n_cen, n_rad, _ = get_center_and_rad(s_pts[input_labels.nonzero().flatten()])    
    
    n_brad = n_rad.item() + context

    m_pts = s_pts - n_cen

    in_inner_rad = (m_pts.norm(dim=1) <= n_rad)
    in_outer_rad = (m_pts.norm(dim=1) <= n_brad)
    
    big_inp_pos_inds = input_labels.nonzero().flatten()
    big_inp_neg_inds = ((~input_labels) & in_outer_rad).nonzero().flatten()    

    MIN_PTS = args.cte_min_reg_pts
    
    if big_inp_pos_inds.shape[0] < MIN_PTS or big_inp_neg_inds.shape[0] < MIN_PTS:
        return None, None

    N = args.cte_part_num_points
    inp_pos_inds, extra_pos_inds = utils.shrink_inds(big_inp_pos_inds, N)
    raw_inp_neg_inds, _ = utils.shrink_inds(big_inp_neg_inds, N)

    # Need this to inp vs gt mean values --> flip more from outside to inside
    if extra_pos_inds is None:
        inp_neg_inds = raw_inp_neg_inds
    else:
        ne = min(extra_pos_inds.shape[0], int(N * NUM_EXTRA_PERC))
        inp_neg_inds = torch.cat((raw_inp_neg_inds[:(N-ne)], extra_pos_inds[:ne]))

    neg_in_inner_perc = in_inner_rad[inp_neg_inds].float().mean().item()
        
    inp_neg_vals = (in_inner_rad[inp_neg_inds].cpu() & (torch.rand(N) > (1-flip_perc))).float()
    inp_pos_vals = (torch.rand(N) > (flip_perc * neg_in_inner_perc)).float()    
    
    comb_inds = torch.cat((inp_pos_inds, inp_neg_inds), dim=0)

    inp_labels = torch.cat((inp_pos_vals, inp_neg_vals), dim =0)
    gt_labels = in_reg[comb_inds]
    inp_pts = s_pts[comb_inds].clone()
    inp_norms = s_norms[comb_inds].clone()

    perc_gt_is_inp = inp_labels[gt_labels.nonzero().flatten()].float().mean().item()
    perc_inp_is_gt = gt_labels[inp_labels.nonzero().flatten()].float().mean().item()
    
    if perc_gt_is_inp <= CHANGE_THRESH or perc_inp_is_gt <= CHANGE_THRESH:
        return None, None

    norm_pts = utils.normalize(inp_pts)

    ex_samps = torch.cat((norm_pts.cpu(), inp_norms.cpu(), inp_labels.view(-1,1)), dim =1)
    ex_labels = gt_labels.float().cpu()            
    
    return ex_samps, ex_labels


