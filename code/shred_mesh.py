import torch
import sys, os
from methods.srd import SRD
import utils

NORMALIZE = True

class Shape:
    def __init__(self, mesh_name):
        verts, faces = utils.loadAndCleanObj(mesh_name)
        verts = torch.tensor(verts).float().to(utils.device)
        faces = torch.tensor(faces).long().to(utils.device)

        if NORMALIZE:
            center = (verts.max(dim=0).values + verts.max(dim=0).values) / 2.
            verts -= center.unsqueeze(0)
            scale = verts.norm(dim=1).max()
            verts /= scale
        
        self.mesh = (verts, faces)
        self.points = None
    

def shred_mesh(mesh_name, out_name):    
    shape = Shape(mesh_name)

    M = SRD()
    srd_points, srd_regions = M.make_pred(shape)

    utils.vis_pc(srd_points, srd_regions, out_name)
        
    
if __name__ == '__main__':
    shred_mesh(sys.argv[1], sys.argv[2])
