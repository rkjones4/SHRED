from eval_method import eval_method, save_data

import utils
import sys
import torch

def main(args):

    method = args.method
    # BASELINES
    if method == 'acd':
        from methods.baselines.acd import ACD
        M = ACD()
    elif method == 'pn_seg':
        from methods.baselines.met_pn_seg import PN_SEG
        M = PN_SEG()
    elif method == 'wopl':
        from methods.baselines.met_wopl import WOPL
        M = WOPL()
    elif method == 'l2g_save':
        from methods.baselines.met_l2g import L2GSave
        print("Saving L2G Inputs to Disk")
        M = L2GSave()
        save_data(M, args.save_name, args.max, args.in_domain_cats, args.out_domain_cats)
        return
    elif method == 'l2g_load':
        from methods.baselines.met_l2g import L2GLoad
        print("Loading L2G Inputs from Disk")
        M = L2GLoad()
        
    # SHRED
    elif method == 'srd':
        from methods.srd import SRD
        M = SRD()

    eval_method(M, args.save_name, args.max, args.in_domain_cats, args.out_domain_cats)


if __name__ == '__main__':
    with torch.no_grad():
        arg_list = [
            ('-mt', '--method', None,  str),        
            ('-en', '--name', None, str),
            ('-svn', '--save_name', None, str),
            ('-mx', '--max', None, int),
            ('-indc', '--in_domain_cats', None, str),
            ('-otdc', '--out_domain_cats', None, str),
        ]                
        
        args = utils.getArgs(arg_list)

        if args.in_domain_cats is not None or args.out_domain_cats is not None:

            if args.in_domain_cats is not None:
                args.in_domain_cats = [c.strip() for c in args.in_domain_cats.split(',') if len(c.strip()) > 0]
            else:
                args.in_domain_cats = []

            if args.out_domain_cats is not None:
                args.out_domain_cats = [c.strip() for c in args.out_domain_cats.split(',') if len(c.strip()) > 0]
            else:
                args.out_domain_cats = []
                
        else:
            args.in_domain_cats = utils.IN_DOMAIN
            args.out_domain_cats = utils.OUT_DOMAIN
            
        if args.save_name is None:
            assert args.name is not None
            args.save_name = args.name
        
        main(args)
