# load 3DTopia-objaverse-caption-361k
import json
import os
from tqdm import tqdm

# load 3DTopia-objaverse-caption-361k
data_path =  "load/3DTopia-objaverse-caption-361k.json"
obj_dir = "datasets/objaverse_debug/exported_json"
save_json = "datasets/objaverse_debug/filtered_3DTopia-objaverse-caption-361k.json"

# load data
data_list = json.load(open(data_path, 'r'))
obj_names = os.listdir(obj_dir)

save_dict = {}

for data in tqdm(data_list):
    obj_name = data['obj_id'] 
    if obj_name in obj_names:
        save_dict[obj_name] = {}
        save_dict[obj_name]['caption'] = data['3dtopia']

# save data
with open(save_json, 'w') as f:
    json.dump(save_dict, f, indent=2)
        
