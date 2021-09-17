"""
1. pre-algin flat cloth to parsed cloth with a tunable parameter "align_factor"
2. obtain palms
3. obtain image gradient
"""
import os
import numpy as np
from PIL import Image
import cv2
import json
import pycocotools.mask as maskUtils
from tqdm import tqdm
import math
import argparse
from matplotlib import pyplot as plt

def get_mask_from_kps(kps,img_h=512,img_w=320):
    rles = maskUtils.frPyObjects(kps, img_h, img_w)
    rle = maskUtils.merge(rles)
    mask = maskUtils.decode(rle)[...,np.newaxis].astype(np.float32)
    mask = mask * 255.0

    return mask


def get_rectangle_mask(a,b,c,d):
    x1 = a + (b-d)/4
    y1 = b + (c-a)/4
    x2 = a - (b-d)/4
    y2 = b - (c-a)/4

    x3 = c + (b-d)/4
    y3 = d + (c-a)/4
    x4 = c - (b-d)/4
    y4 = d - (c-a)/4
    kps = [x1,y1,x2,y2]

    v0_x = c-a
    v0_y = d-b
    v1_x = x3-x1
    v1_y = y3-y1
    v2_x = x4-x1
    v2_y = y4-y1

    cos1 = (v0_x*v1_x+v0_y*v1_y) / (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v1_x*v1_x+v1_y*v1_y))
    cos2 = (v0_x*v2_x+v0_y*v2_y) / (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v2_x*v2_x+v2_y*v2_y))

    if cos1<cos2:
        kps.extend([x3,y3,x4,y4])
    else:
        kps.extend([x4,y4,x3,y3])

    kps = np.array(kps).reshape(1,-1).tolist()
    mask = get_mask_from_kps(kps)

    return mask


def get_hand_mask(hand_keypoints):
    # shoulder, elbow, wrist
    s_x,s_y,s_c = hand_keypoints[0]
    e_x,e_y,e_c = hand_keypoints[1]
    w_x,w_y,w_c = hand_keypoints[2]

    # up_mask = np.ones((256,192,1))
    # bottom_mask = np.ones((256,192,1))
    # up_mask = np.ones((512,512,1))
    # bottom_mask = np.ones((512,512,1))
    up_mask = np.ones((512,320,1))
    bottom_mask = np.ones((512,320,1))
    if s_c > 0.1 and e_c > 0.1:
        up_mask = get_rectangle_mask(s_x,s_y,e_x,e_y)
        kernel = np.ones((20,20),np.uint8)  
        up_mask = cv2.dilate(up_mask,kernel,iterations = 1)
        up_mask = (up_mask > 0).astype(np.float32)[...,np.newaxis]
    if e_c > 0.1 and w_c > 0.1:
        bottom_mask = get_rectangle_mask(e_x,e_y,w_x,w_y)
        bottom_mask = (bottom_mask > 0).astype(np.float32)

    return up_mask, bottom_mask


def get_palm_mask(hand_mask, hand_up_mask, hand_bottom_mask):
    inter_up_mask = (hand_mask + hand_up_mask == 2.0).astype(np.float32)
    inter_bottom_mask = (hand_mask + hand_bottom_mask == 2.0).astype(np.float32)
    palm_mask = hand_mask - inter_up_mask - inter_bottom_mask

    return palm_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--MPV3D_root', type=str, default='path/to/the/MPV3D/dataset',help='path to the MPV3D dataset')
    opt, _ = parser.parse_known_args()

    MPV3D_root = opt.MPV3D_root

    # source dirs
    person_root = os.path.join(MPV3D_root, 'image')
    pose_root = os.path.join(MPV3D_root, 'pose')
    cloth_root= os.path.join(MPV3D_root, 'cloth')
    cloth_mask_root = os.path.join(MPV3D_root, 'cloth-mask')
    parse_root = os.path.join(MPV3D_root, 'image-parse')

    # target dirs
    palmrgb_dst = os.path.join(MPV3D_root, 'palm-rgb')
    palmmask_dst = os.path.join(MPV3D_root, 'palm-mask')
    gradient_dst = os.path.join(MPV3D_root, 'image-sobel')
    os.makedirs(palmrgb_dst, exist_ok=True)
    os.makedirs(palmmask_dst, exist_ok=True)
    os.makedirs(gradient_dst, exist_ok=True)


    # -------------------- Pre-Alignment ------------------------ #
    data_modes = ['train_pairs','test_pairs']
    for data_mode in data_modes:
        # target dirs
        cloth_align_dst = os.path.join(MPV3D_root, 'aligned', data_mode, 'cloth')
        clothmask_align_dst = os.path.join(MPV3D_root, 'aligned', data_mode, 'cloth-mask')
        os.makedirs(cloth_align_dst, exist_ok=True)
        os.makedirs(clothmask_align_dst, exist_ok=True)

        align_factor = 1.0
        p_names, c_names = [], []
        with open(os.path.join(MPV3D_root, data_mode + '.txt'), 'r') as f:
            for line in f.readlines():
                p_name, c_name = line.strip().split()
                p_names.append(p_name)
                c_names.append(c_name)

        for i, imname in tqdm(enumerate(p_names)):
            cname = c_names[i]
            cmname = cname.replace('.jpg','_mask.jpg')
            c_path = os.path.join(cloth_root, cname)
            cm_path = os.path.join(cloth_mask_root, cmname)
            parsename = imname.replace('.png','_label.png')
            parse_pth = os.path.join(parse_root, parsename)
            
                    
            c = Image.open(c_path)
            cm = Image.open(cm_path)
            c_array = np.array(c)
            cm_array = np.array(cm)
            parse = Image.open(parse_pth)
            parse_array = np.array(parse)
            parse_roi = (parse_array == 14).astype(np.float32) + \
                    (parse_array == 15).astype(np.float32) + \
                    (parse_array == 5).astype(np.float32)
            

            # flat-cloth forground & bbox
            c_fg = np.where(cm_array!=0)
            t_c,b_c = min(c_fg[0]), max(c_fg[0])
            l_c,r_c = min(c_fg[1]), max(c_fg[1])
            c_bbox_center = [(l_c+r_c)/2, (t_c+b_c)/2]
            c_bbox_h = b_c - t_c
            c_bbox_w = r_c - l_c

            # parse-cloth forground & bbox
            parse_roi_fg = np.where(parse_roi!=0)
            t_parse_roi, b_parse_roi = min(parse_roi_fg[0]), max(parse_roi_fg[0])
            l_parse_roi, r_parse_roi = min(parse_roi_fg[1]), max(parse_roi_fg[1])
            parse_roi_center = [(l_parse_roi+r_parse_roi)/2, (t_parse_roi+b_parse_roi)/2]
            parse_roi_bbox_h = b_parse_roi - t_parse_roi
            parse_roi_bbox_w = r_parse_roi - l_parse_roi
            
            # scale_factor & paste location
            if c_bbox_w/c_bbox_h > parse_roi_bbox_w/parse_roi_bbox_h:
                ratio = parse_roi_bbox_h / c_bbox_h
                scale_factor = ratio * align_factor
            else:
                ratio = parse_roi_bbox_w / c_bbox_w
                scale_factor = ratio * align_factor
            paste_x = int(parse_roi_center[0] - c_bbox_center[0]*scale_factor)
            paste_y = int(parse_roi_center[1] - c_bbox_center[1]*scale_factor)

            # cloth alignment
            c = c.resize((int(c.size[0]*scale_factor), int(c.size[1]*scale_factor)), Image.BILINEAR)
            blank_c = Image.fromarray(np.ones((512,320,3), np.uint8) * 255)
            blank_c.paste(c, (paste_x, paste_y))
            c = blank_c # PIL Image
            c.save(os.path.join(cloth_align_dst, cname))

            # cloth mask alignment
            cm = cm.resize((int(cm.size[0]*scale_factor), int(cm.size[1]*scale_factor)), Image.NEAREST)
            blank_cm = Image.fromarray(np.zeros((512,320), np.uint8))
            blank_cm.paste(cm, (paste_x, paste_y))
            cm = blank_cm # PIL Image
            cm.save(os.path.join(clothmask_align_dst, cmname))
    print(f'clothes pre-alignment done and saved to {cloth_align_dst}!')


    # -------------------- Segment Palms ------------------------ #
    person_list = sorted(os.listdir(person_root))
    for person_name in tqdm(person_list):
        person_id = person_name.split('_')[0]
        person_path = os.path.join(person_root, person_name)
        parsing_path = os.path.join(parse_root, person_name.replace('.png', '_label.png'))
        keypoints_path = os.path.join(pose_root, person_name.replace('.png', '_keypoints.json'))
        palmrgb_outfn = os.path.join(palmrgb_dst, person_name.replace('whole_front.png','palm.png'))
        palmmask_outfn = os.path.join(palmmask_dst, person_name.replace('whole_front.png','palm_mask.png'))

        parsing = np.array(Image.open(parsing_path))
        person = cv2.imread(person_path).astype(np.float32)

        left_arm_mask = (parsing==14).astype(np.float32)
        right_arm_mask = (parsing==15).astype(np.float32)

        left_arm_mask = np.expand_dims(left_arm_mask, 2)
        right_arm_mask = np.expand_dims(right_arm_mask, 2)

        with open(keypoints_path) as f:
            datas = json.load(f)

        keypoints = np.array(datas['people'][0]['pose_keypoints_2d']).reshape((-1,3))
        left_hand_keypoints = keypoints[[5,6,7],:]
        right_hand_keypoints = keypoints[[2,3,4],:]

        left_hand_up_mask, left_hand_botton_mask = get_hand_mask(left_hand_keypoints)
        right_hand_up_mask, right_hand_botton_mask = get_hand_mask(right_hand_keypoints)

        left_palm_mask = get_palm_mask(left_arm_mask, left_hand_up_mask, left_hand_botton_mask)
        right_palm_mask = get_palm_mask(right_arm_mask, right_hand_up_mask, right_hand_botton_mask)

        palm_rgb = person * (left_palm_mask + right_palm_mask)
        palm_mask = (left_palm_mask + right_palm_mask) * 255

        cv2.imwrite(palmrgb_outfn, palm_rgb)
        cv2.imwrite(palmmask_outfn, palm_mask)
    print(f'palms segmentaion done and saved to {palmmask_dst}!')


    # -------------------- Person Image Gradient ------------------------ #
    person_list = sorted(os.listdir(person_root))
    for person_name in tqdm(person_list):
        person_path = os.path.join(person_root, person_name)
        person = cv2.imread(person_path,0)
        sobelx = cv2.Sobel(person,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(person,cv2.CV_64F,0,1,ksize=5)
        gradientx_outfn = os.path.join(gradient_dst, person_name.replace('.png', '_sobelx.png'))
        gradienty_outfn = os.path.join(gradient_dst, person_name.replace('.png', '_sobely.png'))
        plt.imsave(gradientx_outfn, sobelx, cmap='gray')
        plt.imsave(gradienty_outfn, sobely, cmap='gray')
    print(f'Getting image sobel done and saving to {gradient_dst}!')


    print('******Data preprocessing done!******')