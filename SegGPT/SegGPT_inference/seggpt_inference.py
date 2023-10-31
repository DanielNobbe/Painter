import os
import argparse

import torch
import torch.nn as nn
import numpy as np
from functools import partial

from seggpt_engine import inference_image, inference_video
from models_seggpt import SegGPT


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def init_dirs(out_dir, ovl_dir):
    # create output directories if they don't exit
    os.makedirs(out_dir, exist_ok=True)
    if ovl_dir is not None:
        os.makedirs(ovl_dir, exist_ok=True)


def get_args_parser():
    parser = argparse.ArgumentParser('SegGPT inference', add_help=False)
    parser.add_argument('--ckpt-path', type=str, help='path to ckpt',
                        default='seggpt_vit_large.pth')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='seggpt_vit_large_patch16_input896x448')
    parser.add_argument('--input-image', type=str, help='path to input image to be tested',
                        default=None)
    parser.add_argument('--input-video', type=str, help='path to input video to be tested',
                        default=None)
    parser.add_argument('--num-frames', type=int, help='number of prompt frames in video',
                        default=0)
    parser.add_argument('--prompt-image', type=str, nargs='+', help='path to prompt image',
                        default=None)
    parser.add_argument('--prompt-target', type=str, nargs='+', help='path to prompt target',
                        default=None)
    parser.add_argument('--seg-type', type=str, help='embedding for segmentation types', 
                        choices=['instance', 'semantic'], default='instance')
    parser.add_argument('--device', type=str, help='cuda or cpu',
                        default='cuda')
    parser.add_argument('--output-dir', type=str, help='path to output folder for output mask',
                        default='./')
    parser.add_argument('--overlay-dir', type=str, help='path to output folder for combined mask and input (for visualising)', default=None)
    return parser.parse_args()


def prepare_model(chkpt_dir, arch='seggpt_vit_large_patch16_input896x448', seg_type='instance', input_size=(448, 448)):
    # build model
    model = SegGPT( #(896, 448) (1792, 896) 
        img_size=(2*input_size[0], input_size[1]), patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        drop_path_rate=0.1, window_size=14, qkv_bias=True,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=(list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11)) + \
                                list(range(12, 14)), list(range(15, 17)), list(range(18, 20)), list(range(21, 23))),
        residual_block_indexes=[], use_rel_pos=True, out_feature="last_feat",
        decoder_embed_dim=64,
        loss_func="smoothl1",
        input_size=input_size)
    model.seg_type = seg_type
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model


if __name__ == '__main__':
    args = get_args_parser()

    device = torch.device(args.device)
    model = prepare_model(args.ckpt_path, args.model, args.seg_type).to(device)
    print('Model loaded.')

    init_dirs(args.output_dir, args.overlay_dir)

    assert args.input_image or args.input_video and not (args.input_image and args.input_video)
    if args.input_image is not None:
        assert args.prompt_image is not None and args.prompt_target is not None

        img_name = os.path.basename(args.input_image)
        out_path = os.path.join(args.output_dir,'.'.join(img_name.split('.')[:-1]) + '.png')
        if args.overlay_dir is not None:
            ovl_path = os.path.join(args.overlay_dir, '.'.join(img_name.split('.')[:-1]) + '.png')
        else:
            ovl_path = None

        inference_image(model, device, args.input_image, args.prompt_image, args.prompt_target, out_path, ovl_path)
    
    if args.input_video is not None:
        assert args.prompt_target is not None and len(args.prompt_target) == 1
        vid_name = os.path.basename(args.input_video)
        out_path = os.path.join(args.output_dir, '.'.join(vid_name.split('.')[:-1]) + '.mp4')
        if args.overlay_dir is not None:
            ovl_path = os.path.join(args.overlay_dir, '.'.join(img_name.split('.')[:-1]) + '.mp4')
        else:
            ovl_path = None

        inference_video(model, device, args.input_video, args.num_frames, args.prompt_image, args.prompt_target, out_path, ovl_path)

    print('Finished.')
