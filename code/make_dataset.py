import ast
import os
import sys
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import json as JSON

USER = None
assert USER is not None, 'please set USER variable to point to partNet dataset'

DATA_DIR = f"/home/{USER}/data/partnet/data_v0/"

VERBOSE = True
MAX_SHAPES = None

def getInds():    
    inds = os.listdir(DATA_DIR)
    inds.sort()
    return inds
                      
# finds info for all parts with part_name and with parent with parent name if given
# returns obj-names associated with part and all ori_ids
def newParseParts(folder, json, m2p, pre=''):    
    name = pre + json['name']

    part_id = json['id']
    for m in json['objs']:
        m2p[m] = (name, part_id)
    
    if 'children' in json:
        for c in json['children']:            
            newParseParts(folder, c, m2p, name + '/')                

        
def parseData(folder, category):
    with open(DATA_DIR+folder+"/result_after_merging.json") as f:
        json = ast.literal_eval(f.readline())[0]

    folder_cat = JSON.load(open(DATA_DIR+folder+'/meta.json'))['model_cat'].lower()
        
    if folder_cat != category:
        assert False, 'wrong-cat'

    m2p = {}
    newParseParts(folder, json, m2p)
    
    
    seen_objs = set(m2p.keys())
    all_objs = set([i.split('.')[0] for i in os.listdir(f'{DATA_DIR}{folder}/objs')])
        
    if seen_objs != all_objs:
        assert False, 'missing-obj'

    # Tuples of (label, mesh)
    parts = []
    
    for region, (sem_label, part_id) in m2p.items():                        
        parts.append((region, sem_label, part_id))

    return parts

def writeData(data, out_folder, shape):
    os.system(f'mkdir {out_folder}/{shape} > /dev/null 2>&1')
    j = {}
    j['regions'] = [f'{DATA_DIR}/{shape}/objs/{region}.obj' for region,_,_ in data]
    j['sem_labels'] = [sem_label for _,sem_label,_ in data]
    j['part_ids'] = [part_id for _,_,part_id in data]
    
    JSON.dump(j, open(f'{out_folder}/{shape}/data.json', 'w'))

    
def format_data(out_folder, category):
    os.system(f'mkdir {out_folder} > /dev/null 2>&1')
                
    all_shapes = getInds()
    
    misses = 0
    count = 1e-8

    seen = os.listdir(out_folder)
    
    for shape in tqdm(all_shapes):        
        
        try:
            data = parseData(shape, category)                        
            count += 1
            
        except Exception as e:
            if 'wrong-cat' in e.args[0]:
                continue

            count += 1

            if 'missing-obj'in e.args[0]:
                misses += 1
                continue
            
            else:
                raise e
            
        writeData(data, out_folder, shape)
                        
    print(f"Misses: {misses} ({(misses * 1.) / count})")
                
    
if __name__ == '__main__':
    out_dir = sys.argv[1]
    cat = sys.argv[2]

    format_data(out_dir, cat)

