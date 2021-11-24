"""Dataset class.

You can specify '--dataset_mode unaligned' to use this dataset.
The class name should be consistent with both the filename and its datamode option.
The filename should be <datamode>_dataset.py
The class name should be <Datamode>Dataset.py
"""
from data.base_dataset import BaseDataset, get_transform
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
import json
import os

class UnalignedMPV3dDataset(BaseDataset):
    """unaligned MPV 3D datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.set_defaults(max_dataset_size=10)  # specify dataset-specific default values
        parser.add_argument('--radius', type=int, default=5, help='radius used when drawing pose keypoints')

        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.isTrain = opt.isTrain
        self.model = opt.model
        self.img_width, self.img_height = opt.img_width, opt.img_height
        self.radius = opt.radius
        if self.model == 'TFM' or self.model == 'DRM':
            self.warproot = opt.warproot
        self.im_names, self.c_names = [], []
        with open(os.path.join(self.dataroot, self.datalist+'.txt'), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                self.im_names.append(im_name)
                self.c_names.append(c_name)

        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        # flat-cloth
        if 'MTM' in self.model:
            c = Image.open(os.path.join(self.dataroot, 'cloth', c_name))
            cm = Image.open(os.path.join(self.dataroot, 'cloth-mask', c_name.replace('.jpg','_mask.jpg')))
        elif 'DRM' in self.model:
            c = Image.open(os.path.join(self.warproot, 'warp-cloth', c_name))
            cm = Image.open(os.path.join(self.warproot, 'warp-mask', c_name.replace('.jpg','_mask.jpg')))
        elif 'TFM' in self.model:
            c = Image.open(os.path.join(self.warproot, 'warp-cloth', c_name))
            cm = Image.open(os.path.join(self.warproot, 'warp-mask', c_name.replace('.jpg','_mask.jpg')))
        else:
            raise NotImplementedError('Dataset [%s] is not implemented' % self.model)
        c = self.transform(c)  # [-1,1]
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array) # [0 or 1]
        cm.unsqueeze_(0)

        # person image
        im = Image.open(os.path.join(self.dataroot, 'image', im_name))
        im = self.transform(im) # [-1, 1]

        # person parse
        # hat=1; hair=2; sunglass=4; shirt=5; dress=6; coats=7; pant=9; 
        # neck=10; scarf=11; face=13; left_arm=14; right_arm=15; 
        # left_leg=16; right_leg=17; left_shoe=18, right_shoe=19
        if self.model == 'MTM':
            parse_name = im_name.replace('.png', '_label.png')
            im_parse = Image.open(os.path.join(self.dataroot, 'image-parse', parse_name))
            parse_array = np.array(im_parse)
            im_mask = torch.from_numpy((parse_array > 0).astype(np.float32)).unsqueeze(0)
            im_parse = torch.from_numpy(parse_array).float().unsqueeze(0)
        else:
            parse_name = im_name.replace('front.png', 'segmt.png')
            im_parse = Image.open(os.path.join(self.warproot, 'segmt', parse_name))
            parse_array = np.array(im_parse)
            im_mask = torch.from_numpy((parse_array > 0).astype(np.float32)).unsqueeze(0)
            im_parse = torch.from_numpy(parse_array).float().unsqueeze(0)

        # shape
        if self.model == 'MTM':
            parse_forground = (parse_array > 0).astype(np.float32)
            parse_shape = Image.fromarray((parse_forground*255).astype(np.uint8))
            parse_shape = parse_shape.resize((self.img_width//16, self.img_height//16), Image.BILINEAR)
            parse_shape = parse_shape.resize((self.img_width, self.img_height), Image.BILINEAR)
            im_shape = self.transform(parse_shape) # [-1,1]
        else:
            im_shape = ''

        # upper cloth
        parse_cloth = (parse_array == 5).astype(np.float32)
        pcm = torch.from_numpy(parse_cloth).unsqueeze(0)
        im_c = im * pcm + (1 - pcm) # [-1,1], fill 1 for other parts

        # warped cloth sobel (gradient)
        if self.model == 'DRM':
            c_sobelx = Image.open(os.path.join(self.warproot, 'warp-cloth-sobel', c_name.replace('.jpg', '_sobelx.png'))).convert('L')
            c_sobely = Image.open(os.path.join(self.warproot, 'warp-cloth-sobel', c_name.replace('.jpg', '_sobely.png'))).convert('L')
            c_sobelx, c_sobely = self.transform(c_sobelx), self.transform(c_sobely)
            parse_arm = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)
            parse_arm_cloth = parse_arm + parse_cloth
            pacm = torch.from_numpy(parse_arm_cloth).unsqueeze(0)
            c_sobelx = c_sobelx * pacm - (1 - pacm) # [-1,1], fill -1 for other parts
            c_sobely = c_sobely * pacm - (1 - pacm) # [-1,1], fill -1 for other parts
        else:
            c_sobelx, c_sobely = '', ''

        # head (exclude neck) & hand & lower body (pants, leg, shoes)
        hand_mask = Image.open(os.path.join(self.dataroot, 'palm-mask', im_name.replace('whole_front.png', 'palm_mask.png')))
        hand_mask = (np.array(hand_mask) > 0).astype(np.float32)
        parse_head = (parse_array == 1).astype(np.float32) + \
                (parse_array == 2).astype(np.float32) + \
                (parse_array == 4).astype(np.float32) + \
                (parse_array == 13).astype(np.float32)
        # parse_neck = (parse_array == 10).astype(np.float32)
        parse_lower = (parse_array == 16).astype(np.float32) + \
                (parse_array == 12).astype(np.float32) + \
                (parse_array == 17).astype(np.float32) + \
                (parse_array == 9).astype(np.float32) + \
                (parse_array == 18).astype(np.float32) + \
                (parse_array == 19).astype(np.float32)
        parse_head_hand_lower = parse_head + hand_mask + parse_lower
        phhlm = torch.from_numpy(parse_head_hand_lower).unsqueeze(0)
        im_hhl = im * phhlm - (1 - phhlm) # [-1,1], fill -1 for other parts
        
        # head (include neck) & arm & lower sobel
        if self.model == 'DRM':
            person_sobelx = Image.open(os.path.join(self.dataroot, 'image-sobel', im_name.replace('.png', '_sobelx.png'))).convert('L')
            person_sobely = Image.open(os.path.join(self.dataroot, 'image-sobel', im_name.replace('.png', '_sobely.png'))).convert('L')
            person_sobelx, person_sobely = self.transform(person_sobelx), self.transform(person_sobely)
            parse_arm = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)
            parse_head_arm_lower = parse_head + parse_arm + parse_lower
            phalm = torch.from_numpy(parse_head_arm_lower).unsqueeze(0)
            imhal_sobelx = person_sobelx * phalm - (1 - phalm) # [-1,1], fill -1 for other parts
            imhal_sobely = person_sobely * phalm - (1 - phalm) # [-1,1], fill -1 for other parts
        else:
            imhal_sobelx, imhal_sobely = '', ''
            
        # im depth (front)
        if self.model == 'MTM' and self.isTrain:
            imfd = np.load(os.path.join(self.dataroot, 'depth', im_name.replace('.png', '_depth.npy')))
            imfd_m = (imfd > 0).astype(np.float32)
            imfd = -1 * (2 * imfd -1) # viewport -> ndc -> world
            imfd = imfd * imfd_m
            imfd = torch.from_numpy(imfd).unsqueeze(0)
        if self.model == 'DRM' or self.model == 'TFM':
            imfd = ''
            imfd_initial = np.load(os.path.join(self.warproot, 'initial-depth', im_name.replace('whole_front.png', 'initial_front_depth.npy')))
            imfd_initial = torch.from_numpy(imfd_initial).unsqueeze(0)
        else:
            imfd = ''
            imfd_initial = ''


        # im depth (back)
        if self.model == 'MTM' and self.isTrain:
            imbd = np.load(os.path.join(self.dataroot, 'depth', im_name.replace('front.png', 'back_depth.npy')))
            imbd = np.flip(imbd, axis = 1) # align with imfd
            imbd_m = (imbd > 0).astype(np.float32)
            imbd = 2 * imbd -1 # viewport -> ndc -> world
            imbd = imbd * imbd_m
            imbd = torch.from_numpy(imbd).unsqueeze(0)
        if self.model == 'DRM':
            imbd = ''
            imbd_initial = np.load(os.path.join(self.warproot, 'initial-depth', im_name.replace('whole_front.png', 'initial_back_depth.npy')))
            imbd_initial = torch.from_numpy(imbd_initial).unsqueeze(0)
        else:
            imbd = ''
            imbd_initial = ''

        # load pose points
        if self.model == 'MTM':
            pose_path = os.path.join(self.dataroot, 'pose', im_name.replace('.png', '_keypoints.json'))
            im_pose_tensor, im_pose_vis = self.load_pose(pose_path)
        else:
            im_pose_tensor, im_pose_vis = '', ''

        # agnostic
        if self.model == 'MTM':
            agnostic = torch.cat([im_shape, im_hhl, im_pose_tensor], 0) # (29,512,320)
        else:
            agnostic = ''

        # grid image
        if self.model == 'MTM':
            im_g = Image.open('grid.png')
            im_g = self.transform(im_g)
        else:
            im_g = ''
        
        result = {
            'c_name':               c_name, 
            'im_name':             im_name,   
            'cloth':                     c,
            'cloth_mask':               cm,
            'cloth_sobelx':       c_sobelx,
            'cloth_sobely':       c_sobely,
            'person':                   im,
            'person_parse':       im_parse,
            'person_mask':         im_mask,
            'person_shape':       im_shape,
            'parse_cloth':            im_c,
            'parse_cloth_mask':        pcm,
            'head_hand_lower':      im_hhl,
            'imhal_sobelx':   imhal_sobelx,
            'imhal_sobely':   imhal_sobely,
            'person_fdepth':          imfd,
            'initial_fdepth': imfd_initial,  
            'person_bdepth':          imbd,
            'initial_bdepth': imbd_initial,
            'pose':            im_pose_vis,
            'agnostic':           agnostic,
            'grid_image':             im_g,   
            }

        return result 
    
    def __len__(self):
        """Return the total number of images."""
        # return len(self.image_paths)
        return len(self.im_names)
