import numpy as np
import PIL.Image as Image
import os
from tqdm import tqdm
import cv2
import argparse


def depth2point(fd, bd, rgb, rgb_back, parse_shape=None, label='gt', out_fn=None):
    if label=='gt':
        fd_m = (fd > 0).astype(np.float32)
        fd = -1.0 * (2.0 * fd - 1.0) # --> world
        fd = fd * fd_m
        if parse_shape is not None:
            fd = fd * parse_shape
        bd = np.flip(bd,axis=1)
        bd_m = (bd > 0).astype(np.float32)
        bd = 2.0 * bd - 1.0 # --> world
        bd = bd * bd_m
        if parse_shape is not None:
            bd = bd * parse_shape
    elif label == 'pred':
        if parse_shape is not None:
            fd = fd * parse_shape
            bd = bd * parse_shape
    else:
        print('ERROR: label must be gt/pred!')
    points = []
    for h in range(512):
        for w in range(320):
            if fd[h, w] == 0.: continue
            color = rgb[h, w]
            Z = fd[h,w]
            X = (w + 95) / 256 - 1
            # gly = 512 - 1 - h # Window origin: top left --> OpenGL origin: bottom left
            # Y = gly / 256.0 - 1.0 # [-1,1], Window --> World
            Y = (512 - 1 - h) / 256 - 1
            points.append("%f %f %f %d %d %d\n" % (X, Y, Z, color[0], color[1], color[2]))
    for h in range(512):
        for w in range(320):
            if bd[h, w] == 0.: continue
            color = rgb_back[h, w]
            Z = bd[h,w]
            X = (w + 95) / 256 - 1
            # gly = 512 - 1 - h # Window origin: top left --> OpenGL origin: bottom left
            # Y = gly / 256.0 - 1.0 # [-1,1], Window --> World
            Y = (512 - 1 - h) / 256 - 1
            points.append("%f %f %f %d %d %d\n" % (X, Y, Z, color[0], color[1], color[2]))
    file = open(out_fn, "w")
    file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    %s
    ''' % (len(points), "".join(points)))
    file.close()


def dilate_rgb(newdepth, pix):
    for _ in range(pix):
        d1 = newdepth[3:, :, :] # down
        d2 = newdepth[:-3, :, :] # up
        d3 = newdepth[:, 3:, :] # right
        d4 = newdepth[:, :-3, :] # left
        newdepth[:-3, :,:] = np.where(newdepth[:-3,:,:] > 0, newdepth[:-3, :,:], d1)
        newdepth[3:, :,:] = np.where(newdepth[3:,:,:] > 0, newdepth[3:, :,:] , d2)
        newdepth[:,:-3,:] = np.where(newdepth[:,:-3,:] > 0, newdepth[:, :-3,:], d3)
        newdepth[:, 3:,:] = np.where(newdepth[:,3:,:] > 0, newdepth[:,3:,:], d4)

    return newdepth

def inpaint_back(rgb, parse):
    inpaint_mask = (parse == 13).astype(np.uint8) * 255 + (parse == 10).astype(np.uint8) * 255
    person_inpainted = cv2.inpaint(rgb,inpaint_mask,3,cv2.INPAINT_TELEA)

    return person_inpainted

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--depth_root', type=str, default='results/aligned/DRM/test_pairs/final-depth',help='path to the DRM result')
    parser.add_argument('--frgb_root', type=str, default='results/aligned/TFM/test_pairs/tryon',help='path to the TFM result (front rgb)')
    # parser.add_argument('--brgb_root', type=str, default='esults/aligned/TFM/test_pairs/tryon',help='path to the painted back rgb image')
    parser.add_argument('--parse_root', type=str, default='mpv3d_example/image-parse',help='path to the parsing image')
    parser.add_argument('--point_dst', type=str, default='results/aligned/pcd/test_pairs',help='path to output dir for point cloud')
    opt, _ = parser.parse_known_args()

    depth_list = sorted(os.listdir(opt.depth_root))
    for fd_name in tqdm(depth_list):
        if 'back' in fd_name: continue
        bd_name = fd_name.replace('front_depth.npy', 'back_depth.npy')
        frgb_name = fd_name.replace('_depth.npy', '.png')
        # brgb_name = fd_name.replace('front_depth.npy', 'back.png')
        parse_name = fd_name.replace('depth.npy', 'label.png')
        fd_path = os.path.join(opt.depth_root, fd_name)
        bd_path = os.path.join(opt.depth_root, bd_name)
        frgb_path = os.path.join(opt.frgb_root, frgb_name)
        # brgb_path = os.path.join(opt.brgb_root, brgb_name)
        parse_path  = os.path.join(opt.parse_root, parse_name)
        os.makedirs(opt.point_dst, exist_ok=True)
        point_out_fn = os.path.join(opt.point_dst, fd_name.replace('_whole_front_depth.npy', '.ply'))

        fd = np.load(fd_path)
        fd = fd - 0.02
        bd = np.load(bd_path)
        rm_idx = (fd - bd) < 0
        fd[rm_idx] = 0
        bd[rm_idx] = 0

        # parse = np.array(Image.open(parse_path))
        parse = cv2.imread(parse_path, 0)
        parse_shape = (parse > 0).astype(np.float64)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        parse_shape_erode = cv2.erode(parse_shape, kernel)

        rgb = np.array(Image.open(frgb_path))
        # rgb = cv2.imread(frgb_path).cvtColor(cv2.COLOR_BGR2RGB)
        rgb_fg = rgb * np.expand_dims(parse_shape_erode,2).astype(np.uint8)
        rgb_dilate = dilate_rgb(rgb_fg, 2)
        # rgb_back = np.array(Image.open(brgb_path))
        rgb_back = inpaint_back(rgb, parse)
        rgb_back *= np.expand_dims(parse_shape_erode,2).astype(np.uint8)
        # rgb_back = rgb_dilate
        rgb_back_dilate = dilate_rgb(rgb_back, 2)
        

        depth2point(fd, bd, rgb, rgb_back, parse_shape=parse_shape, label='pred', out_fn=point_out_fn)

    print(f'The unprojected point cloud file(s) are saved to {opt.point_dst}')
    
