# import os
# import glob
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('dir', default='workspace', type=str)
# parser.add_argument('--gpu', default=0, type=int, help='ID of GPU to use')
# parser.add_argument('--save_dir', default='ours_dreamfusion_objs', type=str)
# args = parser.parse_args()

# files = glob.glob(f'{args.dir}/*/*.obj')
# os.makedirs(args.save_dir, exist_ok=True)

# for file in files:
#     name = file.split('/')[-2]
#     save_path = os.path.join(args.save_dir, name)
#     os.system(f"CUDA_VISIBLE_DEVICES={args.gpu} kire {file} --front_dir '\+y' --save {save_path} --wogui --num_azimuth 4 --H 512 --W 512 --elevation '-15' ")

import os
import glob
import argparse
from multiprocessing import Pool

def process_file(args):
    file, gpu_id, save_dir, num_views = args
    name = file.split('/')[-2]
    save_path = os.path.join(save_dir, name)
    command = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} kire {file} --front_dir '\\+y' "
        f"--save {save_path} --wogui --num_azimuth {num_views} --H 512 --W 512 --elevation '-15' --force_cuda_rast"
    )
    os.system(command)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', default='workspace', type=str)
    parser.add_argument('--gpus', default='0', type=str, help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--save_dir', default='ours_dreamfusion_objs', type=str)
    parser.add_argument('--num_views', default=4, type=int)
    args = parser.parse_args()

    files = glob.glob(f'{args.dir}/*/*.obj')
    os.makedirs(args.save_dir, exist_ok=True)

    # Parse GPU IDs
    gpu_ids = args.gpus.split(',')

    # Create a list of arguments for each file, cycling through the GPU IDs
    tasks = [(file, gpu_ids[i % len(gpu_ids)], args.save_dir, args.num_views) for i, file in enumerate(files)]

    with Pool() as pool:
        pool.map(process_file, tasks)