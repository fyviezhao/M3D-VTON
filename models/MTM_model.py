"""Model class.

You can specify '--model MTM' to use this model.
It implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
"""
import torch
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
import sys
sys.path.append("..")
from util import util
import PIL.Image as Image
import numpy as np

class MTMModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.set_defaults(dataset_mode='unaligned')  # You can rewrite default values for this model.
        parser.add_argument('--add_tps', action='store_true', default = True, help='if specified, add tps transformation')
        parser.add_argument('--add_depth', action='store_true', default = True, help='if specified, add depth decoder')
        # parser.add_argument('--use_featB_for_depth', action='store_true', help='if specified, use featB for depth')
        parser.add_argument('--add_segmt', action='store_true', default = True, help='if specified, add segmentation decoder')        
        parser.add_argument('--grid_size', type=int, default=3, help='size of the grid used to estimate TPS params.')                
        parser.add_argument('--input_nc_A', type=int, default=29, help='input nc of feature extraction A [11 for roi agnostic type | 29 for full agnostic type]')
        parser.add_argument('--input_nc_B', type=int, default=3, help='input nc of feature extraction B')
        parser.add_argument('--n_layers_feat_extract', type=int, default=3, help='# layers in feater extraction of MTM')
        parser.add_argument('--add_theta_loss', action='store_true', help='if specified, add theta loss')
        parser.add_argument('--add_grid_loss', action='store_true', help='if specified, add grid loss') 
        # parser.add_argument('--add_inlier_loss', action='store_true', help='if specified, add inlier loss') 
        # parser.add_argument('--dilation_filter', type=int, default=0, help='type of dilation filter [0: no filter | 1: 4-neighs | 2: 8-neighs]')
        parser.add_argument('--lambda_depth', type=float, default=1.0, help='weight of warp loss')
        parser.add_argument('--lambda_segmt', type=float, default=1.0, help='weight of warp loss')
        parser.add_argument('--lambda_warp', type=float, default=1.0, help='weight of warp loss')
        parser.add_argument('--lambda_theta', type=float, default=0.1, help='weight of theta loss')
        parser.add_argument('--lambda_grid', type=float, default=1.0, help='weight of grid loss')
        # parser.add_argument('--lambda_inlier', action='store_true', help='weight of grid loss')
        
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        self.add_tps = opt.add_tps
        self.add_depth = opt.add_depth
        self.add_segmt = opt.add_segmt
        self.use_theta_loss = opt.add_theta_loss
        self.use_grid_loss = opt.add_grid_loss

        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['mtm']
        if self.add_depth:
            self.loss_names.extend(['fdepth', 'bdepth'])
        if self.add_segmt:
            self.loss_names.append('segmt')
        if self.add_tps:
            self.loss_names.append('warp')
        if self.use_theta_loss:
            self.loss_names.append('theta')
        if self.use_grid_loss:
            self.loss_names.append('grid')
              
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['im_hhl','im_shape','pose']
        if self.add_tps:
            self.visual_names.extend(['c', 'tps_cloth', 'tps_grid','im_c','warped_overlay','im'])
        if self.add_segmt:
            self.visual_names.extend(['cm', 'segmt_pred_vis', 'segmt_gt_vis'])
        if self.add_depth:
            self.visual_names.extend(['fdepth_pred', 'fdepth_gt', 'fdepth_diff', 'bdepth_pred', 'bdepth_gt', 'bdepth_diff'])

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks. (you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.)
        self.model_names = ['MTM']
        
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.netMTM = networks.define_MTM(opt.input_nc_A, opt.input_nc_B, opt.ngf, opt.n_layers_feat_extract, opt.img_height, opt.img_width, opt.grid_size, opt.add_tps, opt.add_depth, opt.add_segmt, opt.norm, opt.use_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # only defined during training time
            # define loss functions.
            self.criterionWarp = torch.nn.L1Loss()
            if self.add_depth:
                self.criterionDepth = torch.nn.L1Loss()
            if self.add_segmt:
                class_weights = torch.ones(20, dtype=torch.float32).to(self.device)
                background_skin_idx = [0,1,2,10,13,14,15]
                class_weights[background_skin_idx] = 1.5 # refer to [SieveNet], increase weights for background and skin
                self.criterionSegmt = torch.nn.CrossEntropyLoss(weight=class_weights)
            if self.use_theta_loss:
                self.criterionTheta = networks.ThetaLoss(grid_size=5, device=self.device)
            if self.use_grid_loss:
                self.criterionGrid = networks.GridLoss(opt.img_height, opt.img_width)

            # define and initialize optimizers. You can define one optimizer for each network.
            self.optimizer = torch.optim.Adam(self.netMTM.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizers = [self.optimizer]

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.im_name = input['im_name']                              # meta info
        self.c_name = input['c_name']                                # meta info                    
        self.agnostic = input['agnostic'].to(self.device)            # for input
        self.c = input['cloth'].to(self.device)                      # for input
        self.im_c =  input['parse_cloth'].to(self.device)            # for ground truth
        self.fdepth_gt = input['person_fdepth'].to(self.device)      # for ground truth
        self.bdepth_gt = input['person_bdepth'].to(self.device)      # for ground truth
        self.segmt_gt = input['person_parse'].long().to(self.device) # for ground truth
        self.cm = input['cloth_mask'].to(self.device)                # for visual
        self.im = input['person'].to(self.device)                    # for visual
        self.im_shape = input['person_shape']                        # for visual
        self.im_hhl = input['head_hand_lower']                       # for visual
        self.pose = input['pose']                                    # for visual
        self.im_g = input['grid_image'].to(self.device)              # for visual
        

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.output = self.netMTM(self.agnostic, self.c)

        if self.output['grid_tps'] is not None:
            self.theta_tps = self.output['theta_tps']
            self.grid_tps = self.output['grid_tps']
            self.warped_cloth_mask = self.tps_cloth_mask = F.grid_sample(self.cm, self.grid_tps, padding_mode='zeros')
            self.warped_cloth = self.tps_cloth = F.grid_sample(self.c, self.grid_tps, padding_mode='border')
            self.tps_grid = F.grid_sample(self.im_g, self.grid_tps, padding_mode='zeros')
            self.warped_overlay = (self.tps_cloth + self.im) * 0.5 # just for visual
        
        if self.output['depth'] is not None:
            self.fdepth_pred, self.bdepth_pred = torch.split(self.output['depth'], [1,1], 1)
            self.fdepth_pred = torch.tanh(self.fdepth_pred)
            self.bdepth_pred = torch.tanh(self.bdepth_pred)
            self.fdepth_diff = self.fdepth_pred - self.fdepth_gt # just for visual
            self.bdepth_diff = self.bdepth_pred - self.bdepth_gt # fust for visual
        
        if self.output['segmt'] is not None:
            self.segmt_pred = self.output['segmt']
            self.segmt_pred_argmax = torch.argmax(F.log_softmax(self.segmt_pred, dim=1), dim=1, keepdim=True)

    def backward(self):
        """Calculate losses, gradients; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_mtm = torch.tensor(0.0, requires_grad=True).to(self.device)

        if self.add_tps:
            self.loss_warp = self.opt.lambda_warp * self.criterionWarp(self.warped_cloth, self.im_c)
            self.loss_mtm += self.loss_warp

        if self.add_depth:
            self.loss_fdepth = self.opt.lambda_depth * self.criterionDepth(self.fdepth_pred, self.fdepth_gt)
            self.loss_bdepth = self.opt.lambda_depth * self.criterionDepth(self.bdepth_pred, self.bdepth_gt)
            self.loss_mtm += (self.loss_fdepth + self.loss_bdepth)
        
        if self.add_segmt:
            self.loss_segmt = self.opt.lambda_segmt * self.criterionSegmt(self.segmt_pred, self.segmt_gt.squeeze(1))
            self.loss_mtm += self.loss_segmt

        if self.use_theta_loss:
            self.loss_theta = self.opt.lambda_theta * self.criterionTheta(self.theta_tps)
            self.loss_mtm += self.loss_theta

        if self.use_grid_loss:
            self.loss_grid = self.opt.lambda_grid * self.criterionGrid(self.grid_tps)
            self.loss_mtm += self.loss_grid


        self.loss_mtm.backward()
    
    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward() # compute (fake) output from G      
        self.optimizer.zero_grad()  # clear network MTM's existing gradients
        self.backward()              # calculate gradients for network MTM
        self.optimizer.step()        # update gradients for network MTM

    def compute_visuals(self):
        """Calculate additional output images for tensorbard visualization"""
        # convert raw scores to 1-channel mask that can be visualized
        # segmt_pred: size (batch_size, 1, 512, 320)
        if self.add_segmt:
            self.segmt_pred_vis = util.decode_labels(self.segmt_pred_argmax)
            self.segmt_gt_vis = util.decode_labels(self.segmt_gt)
