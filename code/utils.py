import torch
import numpy as np
import random
import sys
import os
import matplotlib.pyplot as plt
import argparse
import faiss
import math
from copy import deepcopy

IN_DOMAIN = ['chair', 'lamp', 'storagefurniture']
OUT_DOMAIN = ['table', 'knife', 'bed', 'faucet', 'display', 'refrigerator', 'earphone']

device = torch.device('cuda')

NUM_EVAL_POINTS = 10000
        
def getSRColor(i, t):
    mi = 25
    ma = 225

    V = (i * 3) // t

    if V == 0:
        ind = 0
    elif V == 1:
        ind = 1
    else:
        ind = 2

    perc = ((i * 3) / t) - V

    color = [mi, mi, mi]
    color[ind] = round(perc * ma + (1-perc) * mi)
    
    return color
    
def region_vis(samps, regions, fn):
    uniq_regs = regions.cpu().unique().tolist()
    total_colors = len(uniq_regs)
    
    with open(fn, 'w') as f:
            
        for i, u in enumerate(uniq_regs):
            r,g,b = getSRColor(i, total_colors)
            
            inds = (regions == u).nonzero().flatten()
        
            for x,y,z in samps[inds]:
                f.write(f'v {x} {y} {z} {r} {g} {b} \n')
                
def writeObj(verts, faces, outfile):
    with open(outfile, 'w') as f:
        for a, b, c in verts:
            f.write(f'v {a} {b} {c}\n')

        for a, b, c in faces:
            f.write(f"f {a+1} {b+1} {c+1}\n")
            
def face_areas_normals(faces, vs):
    face_normals = torch.cross(
        vs[:, faces[:, 1], :] - vs[:, faces[:, 0], :],
        vs[:, faces[:, 2], :] - vs[:, faces[:, 1], :],
        dim=2,
    )
    face_areas = torch.norm(face_normals, dim=2) + 1e-8
    face_normals = face_normals / face_areas[:, :, None]
    face_areas = 0.5 * face_areas
    return face_areas, face_normals


def sample_surface(faces, vs, count):
    
    if torch.isnan(faces).any() or torch.isnan(vs).any():
        assert False, 'saw nan in sample_surface'

    device = vs.device
    bsize, nvs, _ = vs.shape
    area, normal = face_areas_normals(faces, vs)
    area_sum = torch.sum(area, dim=1)

    assert not (area <= 0.0).any().item(
    ), "Saw negative probability while sampling"
    assert not (area_sum <= 0.0).any().item(
    ), "Saw negative probability while sampling"
    assert not (area > 1000000.0).any().item(), "Saw inf"
    assert not (area_sum > 1000000.0).any().item(), "Saw inf"

    dist = torch.distributions.categorical.Categorical(
        probs=area / (area_sum[:, None]))
    
    face_index = dist.sample((count,))
    keep_face_index = face_index.clone()
    
    # pull triangles into the form of an origin + 2 vectors
    tri_origins = vs[:, faces[:, 0], :]
    tri_vectors = vs[:, faces[:, 1:], :].clone()
    tri_vectors -= tri_origins.repeat(
        1,
        1,
        2
    ).reshape((bsize, len(faces), 2, 3))

    # pull the vectors for the faces we are going to sample from
    face_index = face_index.transpose(0, 1)
    face_index = face_index[:, :, None].expand((bsize, count, 3))
    tri_origins = torch.gather(tri_origins, dim=1, index=face_index)
    face_index2 = face_index[:, :, None, :].expand((bsize, count, 2, 3))
    tri_vectors = torch.gather(tri_vectors, dim=1, index=face_index2)

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = torch.rand(
        count, 2, 1, device=vs.device, dtype=tri_vectors.dtype)

    # points will be distributed on a quadrilateral if we use 2x [0-1] samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(dim=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = torch.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths[None, :]).sum(dim=2)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    normals = torch.gather(normal, dim=1, index=face_index)[0]
    
    return samples[0], keep_face_index.squeeze(), normals


def eval_sample(regions, part_ids, dummy_labels, rd_seed=42, num_points = NUM_EVAL_POINTS):
    random.seed(rd_seed)
    np.random.seed(rd_seed)
    torch.manual_seed(rd_seed)

    verts = []
    faces = []
    segments = []
    labels = []
        
    offset = 0
        
    for ((_v,_f), si, sl) in zip(regions, part_ids, dummy_labels):
        v = torch.from_numpy(_v)
        f = torch.from_numpy(_f)
        faces.append(f + offset)
        verts.append(v)
        segments += [si for _ in range(f.shape[0])]
        labels += [sl for _ in range(f.shape[0])]
        offset += v.shape[0]
            
    verts = torch.cat(verts, dim=0).float().to(device)
    faces = torch.cat(faces, dim=0).long().to(device)
    segments = torch.tensor(segments).long()
    labels = torch.tensor(labels).long()
        
    gpu_samps, gpu_face_inds, gpu_normals = sample_surface(
        faces, verts.unsqueeze(0), num_points
    )
    samps = gpu_samps.cpu()
    face_inds = gpu_face_inds.cpu()
    normals = gpu_normals.cpu()
                    
    samp_segments = segments[face_inds]    
    samp_labels = labels[face_inds]
        
    return (verts, faces), samps, normals, samp_segments, samp_labels

def train_sample(regions, part_ids, num_points = NUM_EVAL_POINTS, rd_seed=None):

    if rd_seed is not None:
        
        random.seed(rd_seed)
        np.random.seed(rd_seed)
        torch.manual_seed(rd_seed)

    verts = []
    faces = []
    segments = []
        
    offset = 0
        
    for ((_v,_f), si) in zip(regions, part_ids):
        v = torch.from_numpy(_v)
        f = torch.from_numpy(_f)
        faces.append(f + offset)
        verts.append(v)
        segments += [si for _ in range(f.shape[0])]
        offset += v.shape[0]
            
    verts = torch.cat(verts, dim=0).float().to(device)
    faces = torch.cat(faces, dim=0).long().to(device)
    segments = torch.tensor(segments).long().to(device)
        
    samps, face_inds, normals = sample_surface(
        faces, verts.unsqueeze(0), num_points
    )
                    
    samp_segments = segments[face_inds]    
        
    return (verts, faces), samps, normals, samp_segments

def loadAndCleanObj(infile):
    raw_verts = []
    raw_faces = []
    seen_verts = set()
    with open(infile) as f:
        for line in f:
            ls = line.split()
            if len(ls) == 0:
                continue
            
            if ls[0] == 'v':

                try:
                    raw_verts.append((
                        float(ls[1]),
                        float(ls[2]),
                        float(ls[3])
                    ))
                except:
                    continue
                
            elif ls[0] == 'f':
                ls = [i.split('//')[0] for i in ls]

                try:
                    raw_faces.append((
                        int(ls[1]),
                        int(ls[2]),
                        int(ls[3]),
                    ))
                except:
                    continue
                
                seen_verts.add(int(ls[1]) -1)
                seen_verts.add(int(ls[2]) -1)
                seen_verts.add(int(ls[3]) -1)

    seen_verts = list(seen_verts)
    seen_verts.sort()
    sv_map = {}
    for i,vn in enumerate(seen_verts):
        sv_map[vn] = i + 1
        
    seen_verts = set(seen_verts)

    verts = []
    faces = []
    for i, v in enumerate(raw_verts):
        if i in seen_verts:
            verts.append(v)

    for face in raw_faces:
        faces.append(
            (
                sv_map[face[0]-1] -1,
                sv_map[face[1]-1] -1,
                sv_map[face[2]-1] -1,
            )
        )

    verts = np.array(verts).astype('float16')
    faces = np.array(faces).astype('long')
    
    return verts, faces

colors = [
    (31, 119, 180),
    (174, 199, 232),
    (255,127,14),
    (255, 187, 120),
    (44,160,44),
    (152,223,138),
    (214,39,40),
    (255,152,150),
    (148, 103, 189),
    (192,176,213),
    (140,86,75),
    (196,156,148),
    (227,119,194),
    (247,182,210),
    (127,127,127),
    (199,199,199),
    (188,188,34),
    (219,219,141),
    (23,190,207),
    (158,218,229)
]

color_names = [
    'royal',
    'sky',
    'orange',
    'goldenrod',
    'forest',
    'lime',
    'red',
    'strawberry',
    'purple',
    'violet',
    'brown',
    'cafe',
    'fuschia',
    'bubblegum',
    'dgrey',
    'lgrey',
    'rio',
    'yellow',
    'aqua',
    'baby'
]

def get_color_name(i):
    ri = i % len(colors)
    num_over = (i // len(colors))
    over = ((num_over + 1) // 2) * 55
    
    sign = 2 * (((num_over+1) % 2 == 0) - .5)    
    delta = over * sign
    raw_name = color_names[ri]

    
    if delta == 0:
        pre = ''
    elif delta < 0:
        pre = 'light '
    elif delta > 0:
        pre = 'dark '

    return pre + raw_name
                
def get_color(i):
    ri = i % len(colors)
    num_over = (i // len(colors))
    over = ((num_over + 1) // 2) * 55
    
    sign = 2 * (((num_over+1) % 2 == 0) - .5)    
    delta = over * sign
    raw_color = colors[ri]    
    return tuple([
        min(max(c+delta,0),255)
        for c in raw_color
    ])

class SuppressStream(object): 

    def __init__(self, stream=sys.stderr):
        self.orig_stream_fileno = stream.fileno()

    def __enter__(self):
        self.orig_stream_dup = os.dup(self.orig_stream_fileno)
        self.devnull = open(os.devnull, 'w')
        os.dup2(self.devnull.fileno(), self.orig_stream_fileno)

    def __exit__(self, type, value, traceback):
        os.close(self.orig_stream_fileno)
        os.dup2(self.orig_stream_dup, self.orig_stream_fileno)
        os.close(self.orig_stream_dup)
        self.devnull.close()


def init_model_run(args, model_type=None):
        
    random.seed(args.rd_seed)
    np.random.seed(args.rd_seed)
    torch.manual_seed(args.rd_seed)

    os.system(f'mkdir {args.outpath} > /dev/null 2>&1')
    
    os.system(f'mkdir {args.outpath}/{args.exp_name} > /dev/null 2>&1')

    if model_type is not None:
        args.exp_name += f'/{model_type}'
        os.system(f'mkdir {args.outpath}/{args.exp_name} > /dev/null 2>&1')

    with open(f"{args.outpath}/{args.exp_name}/config.txt", "w") as f:
        f.write(f'CMD: {" ".join(sys.argv)}\n')        
        f.write(f"ARGS: {args}\n")            

    if model_type is None:
        print("Warning: No Model Type")
        return
        
    os.system(f'mkdir {args.outpath}/{args.exp_name}/plots > /dev/null 2>&1')
    os.system(f'mkdir {args.outpath}/{args.exp_name}/plots/train > /dev/null 2>&1')
    os.system(f'mkdir {args.outpath}/{args.exp_name}/plots/eval > /dev/null 2>&1')        
    os.system(f'mkdir {args.outpath}/{args.exp_name}/models > /dev/null 2>&1')

def log_print(s, args, fn='log'):
    of = f"{args.outpath}/{args.exp_name}/{fn}.txt"
    try:
        with open(of, 'a') as f:
            f.write(f"{s}\n")
        print(s)
    except Exception as e:
        print(f"Failed log print with {e}")

def print_results(
    LOG_INFO,
    result,
    args
):
    res = ""
    for info in LOG_INFO:
        if len(info) == 3:
            name, key, norm_key = info
            if key not in result:
                continue
            _res = result[key] / (result[norm_key]+1e-8)
                
        elif len(info) == 5:
            name, key1, norm_key1, key2, norm_key2 = info
            if key1 not in result or key2 not in result:
                continue
            res1 = result[key1] / (result[norm_key1]+1e-8)
            res2 = result[key2] / (result[norm_key2]+1e-8)
            _res = (res1 + res2) / 2
                
        else:
            assert False, f'bad log info {info}'
                                     
        res += f"    {name} : {round(_res, 4)}\n"

    log_print(res, args)

def make_plots(
    LOG_INFO,
    results,
    plots,
    epochs,
    args,
    fname
):
    for info in LOG_INFO:
        
        for rname, result in results.items():
            if len(info) == 3:
                name, key, norm_key = info
                if key not in result:
                    continue
                res = result[key] / (result[norm_key]+1e-8)
                
            elif len(info) == 5:
                name, key1, norm_key1, key2, norm_key2 = info
                if key1 not in result or key2 not in result:
                    continue
                res1 = result[key1] / (result[norm_key1]+1e-8)
                res2 = result[key2] / (result[norm_key2]+1e-8)
                res = (res1 + res2) / 2
                
            else:
                assert False, f'bad log info {info}'
                        
            if name not in plots[rname]:
                plots[rname][name] = [res]
            else:
                plots[rname][name].append(res)


        plt.clf()
        save = False
        for key in plots:
            if name not in plots[key]:
                continue
            save = True
            plt.plot(
                epochs,
                plots[key][name],
                label= key
            )

        if not save:
            continue
        
        plt.legend()
        plt.grid()
        plt.savefig(f'{args.outpath}/{args.exp_name}/plots/{fname}/{name}.png')


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    v= (cumsum[N:] - cumsum[:-N]) / float(N)

    y = deepcopy(x)
    
    y[N-1:] = v

    return y
    

def make_ravg_plots(
    LOG_INFO,
    results,
    plots,
    epochs,
    args,
    fname
):
    for info in LOG_INFO:
        
        for rname, result in results.items():
            if len(info) == 3:
                name, key, norm_key = info
                if key not in result:
                    continue
                res = result[key] / (result[norm_key]+1e-8)
                
            elif len(info) == 5:
                name, key1, norm_key1, key2, norm_key2 = info
                if key1 not in result or key2 not in result:
                    continue
                res1 = result[key1] / (result[norm_key1]+1e-8)
                res2 = result[key2] / (result[norm_key2]+1e-8)
                res = (res1 + res2) / 2
                
            else:
                assert False, f'bad log info {info}'
                        
            if name not in plots[rname]:
                plots[rname][name] = [res]
            else:
                plots[rname][name].append(res)


        plt.clf()
        save = False
        for key in plots:
            if name not in plots[key]:
                continue
            save = True

            values = plots[key][name]
            
            plt.plot(
                epochs,
                running_mean(values, args.ravg_window),
                label= key
            )

        if not save:
            continue
        
        plt.legend()
        plt.grid()
        plt.savefig(f'{args.outpath}/{args.exp_name}/plots/{fname}/{name}.png')
        
def getArgs(arg_list):       

    parser = argparse.ArgumentParser()


    
    for s,l,d,t in arg_list:        
        parser.add_argument(s, l, default=d, type = t)

    #args = parser.parse_args()
    args, _ = parser.parse_known_args()
    
    return args


def vis_parts(mesh_name, parts):
    face_offset = 1
    
    o_verts = []
    o_faces = []

    for (verts, faces, l) in parts:
        _fo = 0

        color = get_color(l)
        for a, b, c in verts:
            o_verts.append(f'v {a} {b} {c} {color[0]} {color[1]} {color[2]}\n')
            _fo += 1

        for a, b, c in faces:
            o_faces.append(f'f {a+face_offset} {b+face_offset} {c+face_offset}\n')
            
        face_offset += _fo

    with open(mesh_name, 'w') as f:
        for v in o_verts:
            f.write(v)
            
        for fa in o_faces:
            f.write(fa)


def loadObj(infile):
    tverts = []
    ttris = []
    with open(infile) as f:
        for line in f:
            ls = line.split()
            if len(ls) == 0:
                continue

            if ls[0] == 'v':
                tverts.append([
                    float(ls[1]),
                    float(ls[2]),
                    float(ls[3])
                ])
                
            elif ls[0] == 'f':
                ttris.append([
                    int(ls[1].split('/')[0])-1,
                    int(ls[2].split('/')[0])-1,
                    int(ls[3].split('/')[0])-1
                ])

    return tverts, ttris

class PCSearch():
    def __init__(self):
        self.dimension = 3
        
    def build_nn_index(self, database):
        index = faiss.IndexFlatL2(self.dimension)        
        index.add(database)
        return index
    
    def do_search(self, query, target, labels):
        
        query = np.ascontiguousarray(query)
        target = np.ascontiguousarray(target)

        index = self.build_nn_index(target)
        _, I = index.search(query, 1)

        return labels[I.reshape(-1)]

pcsearch = PCSearch()

def clean_regions(segments, max_regs=None):
    
    U = segments.cpu().unique().tolist()

    US = []

    for u in U:
        inds = (segments == u).nonzero().flatten()
        US.append((inds.shape[0], u, inds))

    US.sort(reverse=True)

    clean = torch.zeros(segments.shape, device = segments.device).long() - 1
    
    for i, (_, _, uinds) in enumerate(US):
        if max_regs is not None:
            i = min(i, max_regs-1)
            
        clean[uinds] = i

    assert (clean >= 0).all()

    return clean

def normalize(samps):

    assert len(samps.shape) == 2

    idim = samps.shape[1]

    if idim > 3:
    
        xyz = samps[:,:3]
        rest = samps[:,3:]

    elif idim == 3:
        xyz = samps

    else:
        assert False, f'Bad input shape to normalize {samps.shape}'
        
    c = (xyz.max(dim=0).values + xyz.min(dim=0).values) / 2

    c_xyz = xyz - c
    norm = torch.norm(c_xyz, dim=1).max()

    n_xyz = c_xyz / (norm + 1e-8)

    if idim > 3:
        return torch.cat((n_xyz, rest), dim =1)    
    else:
        return n_xyz
        
    
def shrink_inds(inds, N):                    
    if inds.shape[0] < N:
        inds = inds.repeat(math.ceil(N/inds.shape[0]))
        return inds[:N], None
    else:
        return inds[:N], inds[N:]


def vis_pc(points, labels, pc_name):
    pc_verts = []

    if points is np.ndarray:
        points = torch.from_np(points)
    if labels is np.ndarray:
        labels = torch.from_np(labels)

    for i in labels.cpu().unique().flatten():
        inds = (labels == i).squeeze().nonzero().squeeze()
        color = get_color(i.item())
        pts = points[inds].view(-1, 3)
        for a,b,c in pts:
            pc_verts.append(f'v {a} {b} {c} {color[0]} {color[1]} {color[2]}\n')

    with open(pc_name, 'w') as f:
        for v in pc_verts:
            f.write(v)


