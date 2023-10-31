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


def main():
    input_image = 'input/images/IMG_0124.JPG'
    save_masks = True

    prompt_cfg_path = 'configs/prompt_config.yaml'
    with open(prompt_cfg_path, 'r') as f:
        prompt_cfg = yaml.safe_load(f)

    roi_prompts = prompt_cfg['roi']


    object_prompts = prompt_cfg['object']

    output_path = 'chain-output'
    os.makedirs(output_path, exist_ok=True)

    # prepare model
    device = 'cuda'
    default_ckpt_path = "seggpt_vit_large.pth"
    default_model = "seggpt_vit_large_patch16_input896x448"
    default_seg_type = "instance"  # TODO: Try semantic
    model = prepare_model(default_ckpt_path, default_model, default_seg_type).to(device)

    # run inference to get roi
    prompt_images = roi_prompts['images']
    prompt_masks = roi_prompts['masks']
    roi_output = os.path.join(output_path, 'roi.png')
    roi_overlay_output = os.path.join(output_path, 'roi_overlay.png')
    roi_mask = inference_image(model, device, input_image, prompt_images, prompt_masks, roi_output, roi_overlay_output, return_mask=True, upscale=False)

    # now convert roi_mask to an image to see it
    # threshold mask
    threshold = 20
    roi_mask = roi_mask.max(axis=-1)  # convert to greyscale
    roi_mask = roi_mask > threshold
    if save_masks:
        roi_mask_img = Image.fromarray(255 * roi_mask.astype(np.uint8), mode='L')
        roi_mask_img.save(os.path.join(output_path, 'roi_mask.png'))

    # run inference to get objects
    prompt_images = object_prompts['images']
    prompt_masks = object_prompts['masks']
    object_output = os.path.join(output_path, 'object.png')
    object_overlay_output = os.path.join(output_path, 'object_overlay.png')
    object_mask = inference_image(model, device, input_image, prompt_images, prompt_masks, object_output, object_overlay_output, return_mask=True, upscale=False)

    # convert object_mask to an image to see it
    threshold = 20
    object_mask = object_mask.max(axis=-1)  # convert to greyscale
    object_mask = (object_mask > threshold) & roi_mask
    if save_masks:
        object_mask_img = Image.fromarray(255 * object_mask.astype(np.uint8), mode='L')
        object_mask_img.save(os.path.join(output_path, 'object_mask.png'))


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
    # TODO: Don't scale back up to original resolution, not necessary for
    # fraction

    print(f"Finished!")



if __name__ == '__main__':
    st = time.process_time()
    wt = time.perf_counter()
    main()

    et = time.process_time()
    ewt = time.perf_counter()

    print(f'Finished in {et-st} seconds, wall time {ewt- wt}')