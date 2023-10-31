import os
import argparse

import torch
import numpy as np

from seggpt_engine import inference_image, inference_video
import models_seggpt

from seggpt_inference import prepare_model

from PIL import Image

import time
import yaml

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def prompted_inference(prompts, run_config):
    # TODO: Run on complete list of images
    prompt_images = prompts['prompt_inputs']
    prompt_masks = prompts['prompt_masks']
    output = os.path.join(run_config['output_path'], 'roi.png')
    overlay_output = os.path.join(run_config['output_path'],
                                      'roi_overlay.png')
    input_images = run_config['input_files']
    input_image = input_images[0]
    return inference_image(run_config['model'], run_config['device'],
                               input_image, prompt_images,
                               prompt_masks, output, overlay_output,
                               return_mask=True, upscale=run_config['upscale'])


def save_mask(mask, output_path, mask_name):
    mask_img = Image.fromarray(255 * mask.astype(np.uint8), mode='L')
    mask_img.save(os.path.join(output_path, f'{mask_name}.png'))


def main(args):
    run_cfg_path = args.config_path
    with open(run_cfg_path, 'r') as f:
        run_cfg = yaml.safe_load(f)

    save_masks = run_cfg['save_masks']
    output_path = run_cfg['output_path']
    device = run_cfg['device']
    ckpt_path = run_cfg['ckpt_path']
    model_type = run_cfg['model']
    seg_type = run_cfg['seg_type']  # TODO: Try semantic
    threshold = run_cfg['threshold']


    os.makedirs(output_path, exist_ok=True)

    prompt_cfg_path = run_cfg['prompt_config']
    with open(prompt_cfg_path, 'r') as f:
        prompt_cfg = yaml.safe_load(f)

    roi_prompts = prompt_cfg['roi']
    object_prompts = prompt_cfg['object']


    # prepare model
    model = prepare_model(ckpt_path, model_type, seg_type).to(device)

    # run inference to get roi
    roi_mask = prompted_inference(roi_prompts, run_cfg)

    # threshold mask
    roi_mask = roi_mask.max(axis=-1)  # convert to greyscale
    roi_mask = roi_mask > threshold

    if save_masks:
        save_mask(roi_mask, output_path, 'roi')

    # run inference to get objects
    object_mask = prompted_inference(object_prompts, run_cfg)
    # convert object_mask to an image to see it
    object_mask = object_mask.max(axis=-1)  # convert to greyscale
    object_mask = (object_mask > threshold) & roi_mask
    if save_masks:
        save_mask(object_mask, output_path, 'object')

    # determine number of pixels that make up roi
    roi_pixels = np.count_nonzero(roi_mask)  # could also just do sum
    print("Number of pixels for the roi:", roi_pixels)

    # determine number of pixels that make up objects
    object_pixels = np.count_nonzero(object_mask)
    print("Number of pixels for objects:", object_pixels)

    # calculate percentage of objects
    object_fraction = object_pixels / roi_pixels

    print("object fraction:", object_fraction)

    # TODO: Only use objects segmented in roi for improved
    # accuracy --> could we add this as an extra prompt?

    print(f"Finished!")



if __name__ == '__main__':
    st = time.process_time()
    wt = time.perf_counter()

    parser = argparse.ArgumentParser('SegGPT Chained Inference', add_help=False)
    parser.add_argument('--config-path', type=str, help="Path to run config file (yaml).")
    args = parser.parse_args()
    
    main(args)

    et = time.process_time()
    ewt = time.perf_counter()

    print(f'Finished in {et-st} seconds, wall time {ewt- wt}')