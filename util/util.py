"""This module contains simple helper functions """
from __future__ import print_function
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array (select the first one in case of batch input) into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)         --  the desired data type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale
            image_numpy = np.squeeze(image_numpy, 0) 
            image_numpy = (image_numpy + 1) / 2.0 * 255.0 # post-processing: scaling
        else: # RGB
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
            
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

# color map
label_colors = [(0,0,0),(128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85),
                (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0), (0,0,255), 
                (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]
# transform
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
def decode_labels(mask, num_classes=20):
    """Decode batch of segmentation masks.
    
    Args:
      mask: size of (batch_size,1,height,width), result of inference after taking argmax.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with RGB images of the same (height,width) as the input. 
    """
    mask = mask.squeeze(1)
    n, h, w = mask.shape
    outputs = np.zeros((n,h,w,3), dtype=np.uint8)
    outputs = torch.zeros((n,3,h,w))
    for i in range(n):
        img = Image.new('RGB', (w,h))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]): # j_: row_id, row
            for k_, k in enumerate(j): # k_: col_id, k: value of mask[i, j, k]
                if k < num_classes: # k: [0, num_classes-1]
                    pixels[k_, j_] = label_colors[k]
        outputs[i] = transform(img)

    # convert to normalized torch tensor for later use on tensorboard
    # tmp = np.max(outputs)
    # outputs /= np.max(outputs) # convert [0,255] to [0.0,1.0]
    # outputs = (outputs - 0.5) / 0.5 # normalize to [-1.0,1.0]
    # outputs = torch.from_numpy(np.rollaxis(outputs, 3, 1))
    
    return outputs


def normalize(array):  # array is torch tensor
    return torch.nn.functional.normalize(array, p=2, dim=3)

# set gloabal tensor to init X,Y only for the first time
flag_XY = True
X = torch.zeros([0]).cuda()
Y = torch.zeros([0]).cuda()

def depth2normal_ortho(depth):
    """convert orthographic depth map to normal map (refer to [NormalGAN]).
    Parameters:
        depth (tensor)  -- a batch of depth map: size (B,1,H,W)

    Return:
        normal (tensor)
    """
    
    global flag_XY, X, Y
    B, _, H, W = depth.shape
    depth = depth[:, 0, :, :]

    if flag_XY:
        Y, X = torch.meshgrid(torch.tensor(range(H)), torch.tensor(range(W)))
        X = X.unsqueeze(0).repeat(B, 1, 1).float().cuda()  # (B,H,W)
        Y = Y.unsqueeze(0).repeat(B, 1, 1).float().cuda()
        flag_XY = False
    
    x_cord = (X + 95) / 256 - 1 # specific to MPV3D
    y_cord = (512 - 1 - Y) / 256 - 1 # specific to MPV3D
    p = torch.stack([x_cord, y_cord, depth], dim=3) # (B,H,W,3)

    # vector of p_3d in west, south, east, north direction
    p_ctr = p[:, 1:-1, 1:-1, :]
    vw = p[:, 1:-1, :-2, :] - p_ctr
    vs = p[:, 2:, 1:-1, :] - p_ctr
    ve = p[:, 1:-1, 2:, :] - p_ctr
    vn = p[:, :-2, 1:-1, :] - p_ctr

    vw_z, vs_z, ve_z, vn_z = vw[...,2], vs[...,2], ve[...,2], vn[...,2]
    corner_w = vw * (vw_z < ve_z).unsqueeze(3).float()
    corner_e = ve * (vw_z > ve_z).unsqueeze(3).float()
    corner_n = vn * (vn_z < vs_z).unsqueeze(3).float()
    corner_s = vs * (vn_z > vs_z).unsqueeze(3).float()

    normal_nw = normalize(torch.cross(corner_n, corner_w)) # (B,H-2,W-2,3)
    normal_ws = normalize(torch.cross(corner_w, corner_s))
    normal_se = normalize(torch.cross(corner_s, corner_e))
    normal_en = normalize(torch.cross(corner_e, corner_n))

    normal = normal_nw + normal_ws + normal_se + normal_en 
    normal[torch.where(torch.all(normal == normal[0, 0, 0], dim=-1))] = torch.tensor([0.0,0.0,1.0]).cuda()
    paddings = (0, 0, 1, 1, 1, 1, 0, 0)
    normal = torch.nn.functional.pad(normal, paddings, 'constant') # (B,H,W,3)

    return normal.permute(0, 3, 1, 2) # (B, 3, H, W)


def ply_from_array(points, faces, output_file):

    num_points = len(points)
    num_triangles = len(faces)

    header = '''ply
    format ascii 1.0
    element vertex {}
    property float x
    property float y
    property float z
    element face {}
    property list uchar int vertex_indices
    end_header\n'''.format(num_points, num_triangles)

    with open(output_file,'w') as f:
        f.writelines(header)
        for item in points:
            f.write("{0:0.6f} {1:0.6f} {2:0.6f}\n".format(item[0], item[1], item[2]))

        for item in faces:
            number = len(item)
            row = "{0}".format(number)
            for elem in item:
                row += " {0} ".format(elem)
            row += "\n"
            f.write(row)

def ply_from_array_color(points, colors, faces, output_file):

    num_points = len(points)
    num_triangles = len(faces)

    header = '''ply
    format ascii 1.0
    element vertex {}
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    element face {}
    property list uchar int vertex_indices
    end_header\n'''.format(num_points, num_triangles)

    with open(output_file,'w') as f:
        f.writelines(header)
        index = 0
        for item in points:
            f.write("{0:0.6f} {1:0.6f} {2:0.6f} {3} {4} {5}\n".format(item[0], item[1], item[2],
                                                        colors[index, 0], colors[index, 1], colors[index, 2]))
            index = index + 1

        for item in faces:
            number = len(item)
            row = "{0}".format(number)
            for elem in item:
                row += " {0} ".format(elem)
            row += "\n"
            f.write(row)

def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w = image_numpy.shape[0], image_numpy.shape[1]

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)

def save_depth(depth_numpy, depth_path):
    """Save a numpy depth map to the disk
    
    Parameters:
        depth_numpy (numpy array) -- input depth array
        depth_path (str)          -- the path of the dpeth map
    """
    
    np.save(depth_path, depth_numpy)

def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def expand_dim(tensor,dim,desired_dim_len):
    sz = list(tensor.size())
    sz[dim]=desired_dim_len
    return tensor.expand(tuple(sz))
