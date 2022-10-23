import json
import utils
import torch
import time

def model_train(loader, net, opt, batch_train_fn):

    if isinstance(net, tuple):
        if opt is None:
            net[0].eval()
            net[1].eval()
        else:        
            net[0].train()
            net[1].train()
    
    else:
        
        if opt is None:
            net.eval()
        else:        
            net.train()
        
    ep_result = {}
    bc = 0.
    for batch in loader:
        bc += 1.

        batch_result = batch_train_fn(batch, net, opt)
        
        for key in batch_result:                        
            if key not in ep_result:                    
                ep_result[key] = batch_result[key]
            else:
                ep_result[key] += batch_result[key]

                
    ep_result['batch_count'] = bc

    return ep_result

def run_train_epoch(
    args, res, net, opt, train_loader, val_loader, TRAIN_LOG_INFO, e,
    batch_train_fn, test_loader=None
):

    do_print = (e+1) % args.print_per == 0
    
    json.dump(res, open(f"{args.outpath}/{args.exp_name}/res.json" ,'w'))

    t = time.time()
    
    if do_print:
        utils.log_print(f"\nEpoch {e}:", args)
            
    train_result = model_train(
        train_loader,
        net,
        opt,
        batch_train_fn
    )

    if do_print:
        with torch.no_grad():
            val_result = None
            test_result = None
            if val_loader is not None:
                val_result = model_train(
                    val_loader,
                    net,
                    None,
                    batch_train_fn
                )
                
            if test_loader is not None:
                test_result = model_train(
                    test_loader,
                    net,
                    None,
                    batch_train_fn
                )
                
            res['train_epochs'].append(e)

            ep_result = {
                'train': train_result,
            }

            if val_result is not None:
                ep_result['val'] = val_result

            if test_result is not None:
                ep_result['test'] = test_result                                
            
            utils.log_print(
                f"Train results: ", args
            )
            
            utils.print_results(
                TRAIN_LOG_INFO,
                train_result,
                args,
            )

            
            if e < 0:
                utils.log_print(
                    f"    Time = {time.time() - t}",
                    args
                )
                return
            
            utils.make_plots(
                TRAIN_LOG_INFO,
                ep_result,
                res['train_plots'],
                res['train_epochs'],
                args,
                'train'
            )
            
            if val_result is not None:
                utils.log_print(
                    f"Val results: ", args
                )
            
                utils.print_results(
                    TRAIN_LOG_INFO,
                    val_result,
                    args,
                )

            if test_result is not None:
                utils.log_print(
                    f"Test results: ", args
                )
            
                utils.print_results(
                    TRAIN_LOG_INFO,
                    test_result,
                    args,
                )

            utils.log_print(
                f"    Time = {time.time() - t}",
                args
            )


def run_eval_epoch(
    args,
    res,
    net,
    eval_data,
    EVAL_LOG_INFO,
    e,
    model_eval_fn,
):
        
    if (e+1) % args.eval_per != 0:
        return False
        
    with torch.no_grad():
        
        net.eval()        
                    
        t = time.time()                

        eval_results = {}

        for key, loader in eval_data:

            eval_results[key] = model_eval_fn(
                args,
                loader,
                net,
                e,
                key,
            )
            
            utils.log_print(
                f"Evaluation {key} set results:",
                args
            )

            utils.print_results(
                EVAL_LOG_INFO,
                eval_results[key],
                args
            )
                        
        utils.log_print(f"Eval Time = {time.time() - t}", args)

        res['eval_epochs'].append(e)
                
        utils.make_plots(
            EVAL_LOG_INFO,
            eval_results,            
            res['eval_plots'],
            res['eval_epochs'],
            args,
            'eval'
        )

    return True
