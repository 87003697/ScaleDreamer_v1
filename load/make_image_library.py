import json
import argparse
import os
import random



# python load/make_image_library.py \
#   /home/zhiyuan_ma/code/ScaleDreamer_v1/datasets/sdxl_3d_animation_v1_7 \
#   /home/zhiyuan_ma/code/ScaleDreamer_v1/load/sdxl_3d_animation_v1_7_image_library.json \
#   --val_split 0.3 --test_split 0.3

IMAGE_FILE_TYPES = ['jpg', 'jpeg', 'png']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=str, help='Directory containing images')
    parser.add_argument('output_file', type=str, help='Output file')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split')
    parser.add_argument('--test_split', type=float, default=0.2, help='Test split')
    args = parser.parse_args()

    # loop through all images in the directory
    image_list = []
    for root, dirs, files in os.walk(args.image_dir):
        for file in files:
            if file.split('.')[-1].lower() in IMAGE_FILE_TYPES:
                path = os.path.join(root, file).replace(args.image_dir + '/', '')
                image_list.append(path)

    # shuffle the image list
    random.shuffle(image_list)

    # split the image list into training and validation
    val_num = int(len(image_list) * args.val_split)
    test_num = int(len(image_list) * args.test_split)
    train_list = image_list[val_num + test_num:]
    val_list = image_list[:val_num]
    test_list = image_list[val_num:val_num + test_num]

    # save the image library
    image_library = {
        'train': train_list,
        'val': val_list,
        'test': test_list
    }

    with open(args.output_file, 'w') as f:
        json.dump(image_library, f, indent=2)

    
