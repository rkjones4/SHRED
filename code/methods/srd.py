import sys, os, torch
sys.path.append('methods')
import utils
from init_split import init_split
import met_split_net as msn
import met_fix_net as mfn
import met_merge_net as mmn
import math
from copy import deepcopy

class SRD:
    def __init__(self):
        self.nets = {}
        self.name = 'srd'

        arg_list = [
            ('-np', '--num_points', 100000, int),
            ('-rd', '--rd_seed', 42, int),
            ('-ebs', '--eval_batch_size', 100, int),            

            ('-srd_oo', '--srd_op_order', 'split,fix,merge', str),

            # INIT PARAMS
            
            ('-ism', '--init_split_mode', 'fps', str),
            ('-inb', '--init_num_blocks', 64, int),            
            
            # SPLIT NET PARAMS
            ('-sn_pth', '--sn_path', 'def_models/split_net.pt', str),
            ('-sn_nppb', '--sn_num_points_per_block', 512, int),            
            ('-sn_nc', '--sn_num_clusters', 10, int),
            ('-mm', '--match_mode', 'hung_os', str), 
            
            # FIX NET PARAMS
            ('-fn_pth', '--fn_path', 'def_models/fix_net.pt', str),            
            ('-fn_ctx', '--fn_context', 0.1, float),
            ('-fn_pnp', '--fn_part_num_points', 2048, int),
            ('-fn_cte_mode', '--fn_cte_mode', 'bal', str),

            # LIK NET PARAMS
            ('-ln_ctx', '--ln_context', 0.1, float),
            ('-ln_pnp', '--ln_part_num_points', 512, int),
            ('-ln_pth', '--ln_path', None, str),
            ('-ln_pm', '--ln_pred_mode', None, str),

            # MERGE NET PARAMS

            ('-mn_ctx', '--mn_context', 0.1, float),
            ('-mn_pnp', '--mn_part_num_points', 512, int),
            ('-mn_pth', '--mn_path', 'def_models/merge_net.pt', str),
            ('-mn_ubn', '--mn_use_bn', 'y', str),
            ('-mn_mt', '--mn_merge_thresh', 0.5, float),
            
            # END NET
        
            ('-sn_ubn', '--sn_use_bn', 'y', str),
            ('-ln_ubn', '--ln_use_bn', 'y', str),
            
            # DUMMY PARAMS
            ('-do', '--dropout', 0.0, float),            

            # How many rounds each operation is repeated
            ('-sn_nr', '--sn_num_rounds', 1, int),
            ('-fn_nr', '--fn_num_rounds', 1, int),
            # some high number so merges only stop when no neighbors are merged
            ('-mnr', '--merge_num_rounds', 100, int),

            # control how neighbors are found
            ('-nbt', '--nb_thresh', 0.025, float),
            ('-maxnb', '--max_nbs', 4, int),
            
        ]

        args = utils.getArgs(arg_list)
        
        args.batch_size = args.eval_batch_size
        self.args = args

        if args.sn_path is None and args.sn_path.lower() != 'none':
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("~~~ WARNING: NO SPLIT NET SET ~~~ ")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            self.split_net = None
        else:
            print(f"Loading SN : {args.sn_path}")
            args.use_bn = args.sn_use_bn
            self.split_net = msn.split_net_load(args)
            
        if args.fn_path is None and args.fn_path.lower() != 'none':
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("~~~ WARNING: NO FIX NET SET ~~~ ")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            self.fix_net = None
        else:
            print(f"Loading FN : {args.fn_path}")
            self.fix_net = mfn.fix_net_load(args)

        self.merge_net = None
        
        if args.mn_path is None:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("~~ WARNING: NO MERGING NET SET ~~ ")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")            
            
        if args.mn_path is not None and args.mn_path.lower() != 'none':
            print(f"Loading Merge Net : {args.mn_path}")
            args.use_bn = args.mn_use_bn
            self.merge_net = mmn.merge_net_load(args)

                    
    def split_step(self):
        if self.split_net is None:
            return

        for _ in range(self.args.sn_num_rounds):
            self.update_step(
                msn.model_split(self.split_net, self.args, self.samps, self.regions)
            )

    def fix_step(self):
        if self.fix_net is None:
            return

        for _ in range(self.args.fn_num_rounds):
            self.update_step(
                mfn.model_fix(self.fix_net, self.args, self.samps, self.regions)
            )

    def merge_step(self):
    
        if self.merge_net is None:
            return

        for _ in range(self.args.merge_num_rounds):
            regions, made_merge = mmn.inf_merge(
                self.merge_net,
                self.samps[:,:3],
                self.samps[:,3:],                
                self.regions,
                self.args
            )

            if not made_merge:
                break
            
            self.update_step(regions)

        
    def update_step(self, regions):        
        self.regions = regions
        
    def make_pred(self, data):

        with torch.no_grad():
            
            self.samps, regions = init_split('mesh', self.args, data.mesh)

            self.update_step(regions)
            
            for op in self.args.srd_op_order.split(','):

                if op == 'split':            
                    self.split_step()
                elif op == 'fix':                
                    self.fix_step()
                elif op == 'merge':                    
                    self.merge_step()
                else:
                    assert False, f'Bad Op {op} in order {self.args.srd_op_order}'

            if data.points is None:
                return self.samps[:,:3].cpu(), self.regions.cpu()
                    
            pred = utils.pcsearch.do_search(
                data.points,
                self.samps[:,:3].cpu(),
                self.regions.cpu()
            )

            return pred

