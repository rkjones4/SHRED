import torch
import utils
from utils import device
from methods.split_net.model import fps
import time

def split_regions_fps(args, data):
    
    try:
        mesh, psamps, normals, segments = utils.train_sample(
            data.regions,
            data.part_ids,
            num_points=args.num_points        
        )
            
    except Exception as e:
        utils.log_print(f"Error in sampling {data.ind} -- {e}", args)
        return None, None, None
    
    c_inds = fps(psamps.unsqueeze(0), args.init_num_blocks)[0].long()
    centers = psamps[c_inds]        

    dist = (psamps.view(-1,1,3) - centers.view(1,-1,3)).norm(dim=2)
    fps_clusters = dist.argmin(dim=1).cpu()
    
    clusters = fps_clusters                

    samps = torch.cat((psamps, normals), dim = 1)

    clusters = clusters.to(device)
    
    return samps, utils.clean_regions(clusters), segments
    

def split_mesh_fps(args, mesh):
        
    psamps, _, normals = utils.sample_surface(
        mesh[1], mesh[0].unsqueeze(0), args.num_points
    )
                
    c_inds = fps(psamps.unsqueeze(0), args.init_num_blocks)[0].long()
    centers = psamps[c_inds]        

    dist = (psamps.view(-1,1,3) - centers.view(1,-1,3)).norm(dim=2)
    clusters = dist.argmin(dim=1).to(device)
            
    samps = torch.cat((psamps, normals), dim = 1)
    
    return samps, utils.clean_regions(clusters)


def init_split(input_type, args, data):

    if input_type == 'mesh':        
        if args.init_split_mode == 'fps':
            return split_mesh_fps(args, data)
    
    elif input_type == 'regions':
        if args.init_split_mode == 'fps':
            return split_regions_fps(args, data)
        
    assert False
