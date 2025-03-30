import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.backends.cudnn as cudnn

import argparse
import numpy as np
from tqdm import tqdm

from utils import post_process_depth, flip_lr, compute_errors
from networks.OCTAFlow_model import OCTAFlowNet
from PIL import Image 
from torchvision import transforms
import matplotlib.pyplot as plt

from pathlib import Path

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

def get_val_imgs_list(dataset_root, val_imgs_txt):
    with open(val_imgs_txt, 'r') as f:
        val_filenames = f.readlines()
    val_imgs_list = []
    for val_filename in val_filenames:
        val_imgs_list.append(os.path.join(dataset_root, val_filename.split()[0]))
    return val_imgs_list

parser = argparse.ArgumentParser(description='OCTAFlow PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--encoder',                   type=str,   help='type of encoder, base07, large07', default='large07')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)


if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

args.dataset = 'oca2odt'
args.max_depth = 66
args.encoder = 'tiny07'
args.checkpoint_path = "../ckpt/awake_fold_0_model"
save_dir = "../pred_val_images/awake_dataset/fold_0/lr_2e-4"
dataset_root = "../datasets_release/awake_dataset"
val_imgs_txt = "../data_splits_k_fold_release/awake_dataset/awake_dataset_filepaths_test_fold_0.txt"
val_imgs_list = get_val_imgs_list(dataset_root, val_imgs_txt)
os.makedirs(save_dir, exist_ok=True)


def inference(model, image_path, save_dir, post_process=False):
    
    if args.dataset == 'oca2odt':
        image = Image.open(image_path)
    else:
        image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

    # crop OCA2ODT dataset.
    if args.dataset == 'oca2odt':
        if image.size == (1000, 500):
            y_start, y_end = 10, 490
            x_start, x_end = 20, 980
            image = image.crop((x_start, y_start, x_end, y_end))
            # depth_gt = depth_gt.crop((x_start, y_start, x_end, y_end))

    image = np.asarray(image, dtype=np.float32) / 255.0
    if args.dataset == 'oca2odt':
        # change OCA img from 1 channel to 3 channel.
        if len(image.shape) == 2:
            image = np.stack((image, image, image), axis=2)
                    
    if args.dataset == 'kitti':
        height = image.shape[0]
        width = image.shape[1]
        top_margin = int(height - 352)
        left_margin = int((width - 1216) / 2)
        image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

    image = torch.from_numpy(image.transpose((2, 0, 1)))
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

    with torch.no_grad():
        image = torch.autograd.Variable(image.unsqueeze(0).cuda())
       
        pred_depth = model(image)
        # pred_depths_r_list, _, _ = model(image)
        # if post_process:
        #     image_flipped = flip_lr(image)
        #     pred_depths_r_list_flipped, _, _ = model(image_flipped)
        #     pred_depth = post_process_depth(pred_depths_r_list[-1], pred_depths_r_list_flipped[-1])
        # else:
        #     pred_depth = pred_depths_r_list[-1]

        pred_depth = pred_depth.cpu().numpy().squeeze()

        if args.dataset == 'kitti':
            plt.imsave('depth.png', np.log10(pred_depth), cmap='magma')
        else:
            if args.dataset == 'oca2odt':
                # plt.imsave('depth.png', pred_depth, cmap='gray')
                # save as 16bit png.
                pred_depth = pred_depth*1000
                pred_depth = pred_depth.astype(np.int32)
                pred_depth_img = Image.fromarray(pred_depth)
                save_name = 'pred_'+ Path(image_path).parents[0].name + '_' + Path(image_path).stem + '.png'
                save_path = os.path.join(save_dir, save_name)
                pred_depth_img.save(save_path)
                
            else:
                plt.imsave('depth.png', pred_depth, cmap='jet')
            
          
def main_worker(args):

    model = OCTAFlowNet(version=args.encoder, inv_depth=False, max_depth=args.max_depth, pretrained=None)
    model.train()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("== Total number of learning parameters: {}".format(num_params_update))

    model = torch.nn.DataParallel(model)
    model.cuda()

    print("== Model Initialized")

    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            print("== Loaded checkpoint '{}'".format(args.checkpoint_path))
            del checkpoint
        else:
            print("== No checkpoint found at '{}'".format(args.checkpoint_path))

    cudnn.benchmark = True

    # ===== Inference ======
    model.eval()
    with torch.no_grad():
        # inference(model, post_process=True)
        for image_path in tqdm(val_imgs_list):
            inference(model, image_path, save_dir, post_process=False)


def main():
    torch.cuda.empty_cache()
    args.distributed = False    
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1:
        print("This machine has more than 1 gpu. Please set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1
    
    main_worker(args)


if __name__ == '__main__':
    main()
