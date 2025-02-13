import os
import cv2
import torch
import time
from PIL import Image
import numpy as np
from tqdm import tqdm
import json

from diffusers import StableDiffusionXLPipeline

from rembg import remove
from segment_anything import sam_model_registry, SamPredictor

import argparse

parser = argparse.ArgumentParser(description='SDXL T2I Custom')
parser.add_argument('--prompt_libary', type=str, default='default', help='prompt library')
parser.add_argument('--group_total', type=int, default=1, help='the total number of groups')
parser.add_argument('--group_index', type=int, default=0, help='the index of the group')
parser.add_argument('--num_repeat', type=int, default=1, help='the number of repeat for each prompt')
parser.add_argument('--save_dir', type=str, default='./imgs/outputs', help='the directory to save the outputs')
args = parser.parse_args()

# load the data
assert os.path.exists(args.prompt_libary), 'Prompt library does not exist'
with open(args.prompt_libary, 'r') as f:
    data_dict = json.load(f)
    all_prompts = \
            data_dict['train'] + \
            data_dict['val'] + \
            data_dict['test']


# use only a part of the prompts
all_prompts =  all_prompts[args.group_index::args.group_total]
# skip long prompts
all_prompts = [p for p in all_prompts if len(p.split(" ")) < 20]
print (f'Prompt count: {len(all_prompts)}')

# copied from Era3D, https://github.com/pengHTYX/Era3D/blob/main/app.py
####################################################################################################
def sam_init():
    sam_checkpoint = os.path.join("./pretrained", "sam_vit_h_4b8939.pth")
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=f"cuda:0")
    predictor = SamPredictor(sam)
    return predictor

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
def sam_segment(predictor, input_image, *bbox_coords):
    bbox = np.array(bbox_coords)
    image = np.asarray(input_image)

    start_time = time.time()
    predictor.set_image(image)

    masks_bbox, scores_bbox, logits_bbox = predictor.predict(box=bbox, multimask_output=True)

    # print(f"SAM Time: {time.time() - start_time:.3f}s")
    out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image
    out_image_bbox = out_image.copy()
    out_image_bbox[:, :, 3] = masks_bbox[-1].astype(np.uint8) * 255
    torch.cuda.empty_cache()
    return Image.fromarray(out_image_bbox, mode='RGBA')

def preprocess(predictor, input_image, chk_group=None, segment=True, rescale=False):
    RES = 512
    input_image.thumbnail([RES, RES], Image.Resampling.LANCZOS)
    if chk_group is not None:
        segment = "Background Removal" in chk_group
        rescale = "Rescale" in chk_group
    if segment:
        image_rem = input_image.convert('RGBA')
        image_nobg = remove(image_rem, alpha_matting=True)
        arr = np.asarray(image_nobg)[:, :, -1]
        x_nonzero = np.nonzero(arr.sum(axis=0))
        y_nonzero = np.nonzero(arr.sum(axis=1))
        x_min = int(x_nonzero[0].min())
        y_min = int(y_nonzero[0].min())
        x_max = int(x_nonzero[0].max())
        y_max = int(y_nonzero[0].max())
        input_image = sam_segment(predictor, input_image.convert('RGB'), x_min, y_min, x_max, y_max)
    else:
        input_image = input_image.convert('RGBA')
    # Rescale and recenter
    if rescale:
        image_arr = np.array(input_image)
        in_w, in_h = image_arr.shape[:2]
        out_res = min(RES, max(in_w, in_h))
        ret, mask = cv2.threshold(np.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(mask)
        max_size = max(w, h)
        ratio = 0.75
        side_len = int(max_size / ratio)
        padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
        center = side_len // 2
        padded_image[center - h // 2 : center - h // 2 + h, center - w // 2 : center - w // 2 + w] = image_arr[y : y + h, x : x + w]
        rgba = Image.fromarray(padded_image).resize((out_res, out_res), Image.LANCZOS)

        rgba_arr = np.array(rgba) / 255.0
        rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
        input_image = Image.fromarray((rgb * 255).astype(np.uint8))
    else:
        input_image = expand2square(input_image, (127, 127, 127, 0))
    return input_image, input_image.resize((320, 320), Image.Resampling.LANCZOS)
####################################################################################################

predictor = sam_init()

pipeline_text2image = StableDiffusionXLPipeline.from_pretrained(
    "pretrained/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda:0")

pipeline_text2image.load_lora_weights(
    "pretrained/3D_Animation_Style-000009.safetensors", 
)
pipeline_text2image.fuse_lora(lora_scale=1.0)



for prompt in tqdm(all_prompts, desc="Generating images at {}".format(args.group_index)):
    if len(prompt.replace(",", " ").split()) > 20:
        continue
    save_dir = os.path.join(args.save_dir, prompt.lower().replace(" ", "_"))
    os.makedirs(save_dir, exist_ok=True)
    images = pipeline_text2image(
        prompt=prompt,
        num_images_per_prompt=args.num_repeat
    ).images
    for i, image in enumerate(images):
        i += 100 # start from 100
        processed_image_highres, processed_image = preprocess(predictor, image, chk_group=None, segment=True, rescale=True)
        processed_image_highres.save(os.path.join(save_dir, f"{i:03d}.png"))
