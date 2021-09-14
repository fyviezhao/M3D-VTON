"""General-purpose test script for M3D-VTON

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.
It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images.

See options/base_options.py and options/test_options.py for more test options.
"""
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.util import tensor2im, save_image, save_depth, decode_labels, depth2normal_ortho
import PIL.Image as Image


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    
    # make destination dirs.
    results_dir = os.path.join(opt.results_dir, opt.datamode, opt.name, opt.datalist)
    os.makedirs(results_dir, exist_ok=True)
    if 'MTM' in opt.model:
        os.makedirs(os.path.join(results_dir, 'warp-cloth'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'warp-mask'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'warp-grid'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'warp-cloth-sobel'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'segmt'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'segmt-vis'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'initial-depth'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'initial-depth-vis'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'initial-normal-vis'), exist_ok=True)
    if 'TFM' in opt.model:
        os.makedirs(os.path.join(results_dir, 'tryon'), exist_ok=True)
    if 'DRM' in opt.model:
        os.makedirs(os.path.join(results_dir, 'final-depth'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'final-depth-vis'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'final-normal-vis'), exist_ok=True)
    
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True     # disable data shuffling; comment this line if results on randomly chosen images are needed.
    dataset = create_dataset(opt) # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)
    model = create_model(opt)     # create a model given opt.model and other options
    model.setup(opt)              # regular setup: load and print networks
  
    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:   # only apply our model to opt.num_test images.
            break
        model.set_input(data)   # unpack data from data loader
        model.test()            # run inference
        im_name = model.im_name[0] # get person name
        c_name = model.c_name[0]   # get cloth name
        # visuals = model.get_current_visuals()  # get image results
        print('processing (%04d)-th / (%04d) image...' % (i+1, dataset_size), end='\r')
        time.sleep(0.001)

        if 'MTM' in opt.model: # save warped_cloth, warped_cloth_mask, warp_grid, roi_segmt and roi_depth to disk
            if opt.add_tps:
                warped_cloth = tensor2im(model.warped_cloth)
                warped_grid = tensor2im(model.tps_grid)
                save_image(warped_cloth, os.path.join(results_dir, 'warp-cloth', c_name))
                save_image(model.warped_cloth_mask.mul(255).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8), os.path.join(results_dir, 'warp-mask', c_name.replace('.jpg','_mask.jpg')))
                save_image(warped_grid, os.path.join(results_dir, 'warp-grid', c_name.replace('.jpg','_grid.jpg')))
                # save cloth sobel
                warped_cloth_gray = cv2.cvtColor(warped_cloth,cv2.COLOR_RGB2GRAY)
                warped_cloth_sobelx = cv2.Sobel(warped_cloth_gray,cv2.CV_64F,1,0,ksize=5)
                warped_cloth_sobely = cv2.Sobel(warped_cloth_gray,cv2.CV_64F,0,1,ksize=5)
                plt.imsave(os.path.join(results_dir, 'warp-cloth-sobel', c_name.replace('.jpg', '_sobelx.png')), warped_cloth_sobelx, cmap='gray')
                plt.imsave(os.path.join(results_dir, 'warp-cloth-sobel', c_name.replace('.jpg', '_sobely.png')), warped_cloth_sobely, cmap='gray')
            if opt.add_depth:
                fdepth_pred = model.fdepth_pred.squeeze(0).squeeze(0).cpu().float().numpy()
                bdepth_pred = model.bdepth_pred.squeeze(0).squeeze(0).cpu().float().numpy()
                save_depth(fdepth_pred, os.path.join(results_dir, 'initial-depth', im_name.replace('whole_front.png', 'initial_front_depth.npy')))
                save_depth(bdepth_pred, os.path.join(results_dir, 'initial-depth', im_name.replace('whole_front.png', 'initial_back_depth.npy')))
                if opt.save_depth_vis:
                    fdepth_pred_vis = (fdepth_pred + 1) / 2.0 * 255.0
                    bdepth_pred_vis = (bdepth_pred + 1) / 2.0 * 255.0
                    save_image(fdepth_pred_vis.astype(np.uint8), os.path.join(results_dir, 'initial-depth-vis', im_name.replace('whole_front.png', 'initial_front_depth.png')))
                    save_image(bdepth_pred_vis.astype(np.uint8), os.path.join(results_dir, 'initial-depth-vis', im_name.replace('whole_front.png', 'initial_back_depth.png')))
                if opt.save_normal_vis:
                    fnormal_pred = depth2normal_ortho(model.fdepth_pred).squeeze(0)
                    fnormal_np = fnormal_pred.permute(1,2,0).cpu().numpy()
                    fnormal_vis = fnormal_np * 0.5 + 0.5
                    fnormal_vis = (fnormal_vis * 255).astype(np.uint8)
                    fnormal_pil = Image.fromarray(fnormal_vis)
                    fnormal_pil.save(os.path.join(results_dir, 'initial-normal-vis', im_name.replace('.png','_normal.png')))
                    bnormal_pred = depth2normal_ortho(model.bdepth_pred).squeeze(0)
                    bnormal_np = bnormal_pred.permute(1,2,0).cpu().numpy()
                    bnormal_vis = bnormal_np * 0.5 + 0.5
                    bnormal_vis = (bnormal_vis * 255).astype(np.uint8)
                    bnormal_pil = Image.fromarray(bnormal_vis)
                    bnormal_pil.save(os.path.join(results_dir, 'initial-normal-vis', im_name.replace('front.png','back_normal.png')))
            if opt.add_segmt:
                save_image(model.segmt_pred_argmax.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8), os.path.join(results_dir, 'segmt', im_name.replace('front.png', 'segmt.png')))
                if opt.save_segmt_vis: # WARNING: very slow
                    save_image(tensor2im(decode_labels(model.segmt_pred_argmax)), os.path.join(results_dir, 'segmt-vis', im_name.replace('front.png', 'segmt_vis.png')))
        
        if 'TFM' in opt.model: # save p_tryon to disk
            save_image(tensor2im(model.p_tryon), os.path.join(results_dir, 'tryon', im_name))
        
        if 'DRM' in opt.model: # save refined depth to disk
            imfd_pred = model.imfd_pred.squeeze(0).squeeze(0).cpu().float().numpy()
            imbd_pred = model.imbd_pred.squeeze(0).squeeze(0).cpu().float().numpy()
            save_depth(imfd_pred, os.path.join(results_dir, 'final-depth', im_name.replace('.png','_depth.npy')))
            save_depth(imbd_pred, os.path.join(results_dir, 'final-depth', im_name.replace('front.png','back_depth.npy')))
            if opt.save_depth_vis:
                imfd_pred_vis = (imfd_pred + 1) / 2.0 * 255.0
                imbd_pred_vis = (imbd_pred + 1) / 2.0 * 255.0
                save_image(imfd_pred_vis.astype(np.uint8), os.path.join(results_dir, 'final-depth-vis', im_name.replace('.png', '_depth.png')))
                save_image(imbd_pred_vis.astype(np.uint8), os.path.join(results_dir, 'final-depth-vis', im_name.replace('front.png', 'back_depth.png')))
            if opt.save_normal_vis:
                fnormal_pred = depth2normal_ortho(model.imfd_pred).squeeze(0)
                fnormal_np = fnormal_pred.permute(1,2,0).cpu().numpy()
                fnormal_vis = fnormal_np * 0.5 + 0.5
                fnormal_vis = (fnormal_vis * 255).astype(np.uint8)
                fnormal_pil = Image.fromarray(fnormal_vis)
                fnormal_pil.save(os.path.join(results_dir, 'final-normal-vis', im_name.replace('.png','_normal.png')))
                bnormal_pred = depth2normal_ortho(model.imbd_pred).squeeze(0)
                bnormal_np = bnormal_pred.permute(1,2,0).cpu().numpy()
                bnormal_vis = bnormal_np * 0.5 + 0.5
                bnormal_vis = (bnormal_vis * 255).astype(np.uint8)
                bnormal_pil = Image.fromarray(bnormal_vis)
                bnormal_pil.save(os.path.join(results_dir, 'final-normal-vis', im_name.replace('front.png','back_normal.png')))
    print(f'\nTest {opt.model} down.')