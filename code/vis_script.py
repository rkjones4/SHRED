import utils
import torch
import numpy as np

MET_MAP = {
    'acd': 'final_acd',
    'fps': 'final_fps',
    'l2g': 'final_l2g',
    'shred_2': 'final_shred_mt_20',
    'shred_5': 'final_shred_mt_50',
    'shred_8': 'final_shred_mt_80',
    'wopl': 'final_wopl',
    'wopl_prior': 'final_wopl_prior',    
    'gt': 'gt'
}


def view(args, method):
    sn = MET_MAP[method]

    all_reg_preds = torch.from_numpy(np.load(f'save_results/{sn}/preds_{args.cat}.npy'))
    all_pts = torch.from_numpy(np.load(f'save_results/points_{args.cat}.npy'))
    
    if args.inds is None:
        inds = list(range(all_reg_preds.shape[0]))
    else:
        inds = [int(i) for i in args.inds.split(',')]

    for ind in inds:
        utils.vis_pc(
            all_pts[ind],
            all_reg_preds[ind],
            f'pred_{method}_{ind}.obj'
        )
    

if __name__ == '__main__':
    
    arg_list = [
        ('-mt', '--method', None,  str),        
        ('-inds', '--inds', None, str),
        ('-cat', '--cat', None, str)
    ]

    args = utils.getArgs(arg_list)

    if args.method is not None:
        view(args, args.method)
    else:
        for method in MET_MAP.keys():
            view(args, method)

    
