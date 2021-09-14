"""Model class.

You can specify '--model DRM' to use this model.
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
from .base_model import BaseModel
from . import networks
import sys
sys.path.append('..')
from util import util


class DRMModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--warproot', type=str, default='results/aligned/MTM/test_pairs', help='path to MTM result folder')
        parser.add_argument('--input_gradient', action='store_true', default=True, help='if specified, add image (sobel) gradient to input')
        parser.add_argument('--input_nc', type=int, default=8, help='input nc for DRM generator [base: 8 | input gradient: + 4]')
        parser.add_argument('--output_nc', type=int, default=2, help='output nc for DRM generator')
        parser.set_defaults(display_ncols=2)  # rewrite default values for this model.
        parser.set_defaults(netD='basic')
        parser.add_argument('--input_nc_D', type=int, default=4, help='3-channel normal map and 1-channel mask')
        parser.add_argument('--add_gan_loss', action='store_true', help='if specified, use gan loss')
        parser.add_argument('--add_grad_loss', action='store_true', default=True, help='if specified, use depth gardient loss')
        parser.add_argument('--add_normal_loss', action='store_true', help='if specified, use normal loss')
        parser.add_argument('--lambda_depth', type=float, default=1.0, help='weight for the depth loss')
        parser.add_argument('--lambda_gan', type=float, default=1.0, help='weight for the gan loss')
        parser.add_argument('--lambda_grad', type=float, default=1.0, help='weight for the depth gradinet loss')
        parser.add_argument('--lambda_normal', type=float, default=1.5, help='weight for the depth gradinet loss')
        
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
        self.input_gradient = opt.input_gradient
        self.use_gan_loss = opt.add_gan_loss
        self.use_grad_loss = opt.add_grad_loss
        self.use_normal_loss = opt.add_normal_loss
        if self.use_grad_loss:
            self.compute_grad = networks.Sobel().to(self.device)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['drm', 'fdepth', 'bdepth']
        if self.use_grad_loss:
            self.loss_names.extend(['fgrad', 'bgrad'])
        if self.use_normal_loss:
            self.loss_names.extend(['fnormal', 'bnormal'])
        if self.use_gan_loss:
            self.loss_names.extend(['fgan','bgan', 'FND', 'BND'])

        
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['c', 'im_hhl', 'imfd_initial', 'imbd_initial']
        if self.input_gradient:
            self.visual_names.extend(['imhal_sobelx', 'imhal_sobely', 'c_sobelx', 'c_sobely'])
        self.visual_names.extend(['imfd_pred','imbd_pred', 'imfd_diff', 'imbd_diff'])
        if self.use_grad_loss:
            self.visual_names.extend(['fgrad_pred_x','fgrad_x', 'fgrad_pred_y', 'fgrad_y', 'fgrad_x_diff', 'fgrad_y_diff', 'bgrad_pred_x', 'bgrad_x', 'bgrad_pred_y', 'bgrad_y',  'bgrad_x_diff', 'bgrad_y_diff'])
        if self.use_normal_loss or self.use_gan_loss:
            self.visual_names.extend(['imfn_pred', 'imfn', 'imbn_pred', 'imbn', 'imfn_diff', 'imbn_diff'])
        
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        if self.isTrain and self.use_gan_loss:
            self.model_names = ['DRM', 'FND', 'BND']
        else:
            self.model_names = ['DRM']
        

        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        if self.input_gradient:
            opt.input_nc += 4
        self.netDRM = networks.define_DRM(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain and self.use_gan_loss: # define front & back normal discriminator
            self.netFND = networks.define_D(opt.input_nc_D, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netBND = networks.define_D(opt.input_nc_D, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.isTrain:  # only defined during training time
            # define loss functions
            self.criterionDepth = networks.DepthLoss().to(self.device)
            if self.use_grad_loss:
                self.criterionGrad = networks.DepthGradLoss().to(self.device)
            if self.use_gan_loss:
                self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            if self.use_normal_loss:
                self.criterionNormal = networks.NormalLoss()

            # define and initialize optimizers. You can define one optimizer for each network.
            self.optimizer_G = torch.optim.Adam(self.netDRM.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizers = [self.optimizer_G]
            if self.use_gan_loss:
                self.optimizer_FND = torch.optim.Adam(self.netFND.parameters(), lr=opt.lr, betas=(0.5, 0.999))
                self.optimizer_BND = torch.optim.Adam(self.netBND.parameters(), lr=opt.lr, betas=(0.5, 0.999))
                self.optimizers.extend([self.optimizer_FND, self.optimizer_BND])

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.im_name = input['im_name']                             # meta info
        self.c_name = input['c_name']                               # meta info
        self.c = input['cloth'].to(self.device)                     # for input
        self.im_hhl = input['head_hand_lower'].to(self.device)      # for input
        self.imfd_initial = input['initial_fdepth'].to(self.device) # for input
        self.imbd_initial = input['initial_bdepth'].to(self.device) # for input
        if self.input_gradient:
            self.imhal_sobelx = input['imhal_sobelx'].to(self.device) # for input 
            self.imhal_sobely = input['imhal_sobely'].to(self.device) # for input
            self.c_sobelx = input['cloth_sobelx'].to(self.device)     # for input
            self.c_sobely = input['cloth_sobely'].to(self.device)     # for input
        self.imfd = input['person_fdepth'].to(self.device)          # for ground truth
        self.imbd = input['person_bdepth'].to(self.device)          # for ground truth
        if self.use_grad_loss:
            self.fgrad = self.compute_grad(self.imfd) # for ground truth
            self.bgrad = self.compute_grad(self.imbd) # for ground truth

        if self.use_normal_loss or self.use_gan_loss:
            self.im_mask = input['person_mask'].to(self.device) # for input
            self.imfn = util.depth2normal_ortho(self.imfd).to(self.device) # for ground truth
            self.imbn = util.depth2normal_ortho(self.imbd).to(self.device) # for ground truth

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        if self.input_gradient:
            self.input = torch.cat([self.imfd_initial, self.imbd_initial, self.c, self.im_hhl, self.c_sobelx, self.c_sobely, self.imhal_sobelx, self.imhal_sobely], 1)
        else:
            self.input = torch.cat([self.imfd_initial, self.imbd_initial, self.c, self.im_hhl], 1)
        outputs = self.netDRM(self.input)
        self.imfd_pred, self.imbd_pred= torch.split(outputs, [1,1], 1)
        self.imfd_pred = torch.tanh(self.imfd_pred)
        self.imbd_pred = torch.tanh(self.imbd_pred)

        if self.use_grad_loss:
            self.fgrad_pred = self.compute_grad(self.imfd_pred)
            self.bgrad_pred = self.compute_grad(self.imbd_pred)

        if self.use_normal_loss or self.use_gan_loss:
            self.imfn_pred = util.depth2normal_ortho(self.imfd_pred)
            self.imbn_pred = util.depth2normal_ortho(self.imbd_pred)

    def backward_G(self):
        """Calculate losses, gradients; called in every training iteration"""
        # losses for generator only
        self.loss_fdepth = self.opt.lambda_depth * self.criterionDepth(self.imfd_pred, self.imfd)
        self.loss_bdepth = self.opt.lambda_depth * self.criterionDepth(self.imbd_pred, self.imbd)
        self.loss_drm = self.loss_fdepth + self.loss_bdepth

        if self.use_grad_loss:
            self.loss_fgrad = self.opt.lambda_grad * self.criterionGrad(self.fgrad_pred, self.fgrad)
            self.loss_bgrad = self.opt.lambda_grad * self.criterionGrad(self.bgrad_pred, self.bgrad)
            self.loss_drm += self.loss_fgrad + self.loss_bgrad

        if self.use_normal_loss:
            self.loss_fnormal = self.opt.lambda_normal * self.criterionNormal(self.imfn_pred, self.imfn)
            self.loss_bnormal = self.opt.lambda_normal * self.criterionNormal(self.imbn_pred, self.imbn)
            self.loss_drm += self.loss_fnormal + self.loss_bnormal

        if self.use_gan_loss: # G(fake_input) should fake the discriminator
            pred_fake_fnormal = self.netFND(torch.cat([self.im_mask,self.imfn_pred], 1))
            pred_fake_bnormal = self.netBND(torch.cat([self.im_mask,self.imbn_pred], 1))
            self.loss_fgan = self.opt.lambda_gan * self.criterionGAN(pred_fake_fnormal, True)
            self.loss_bgan = self.opt.lambda_gan * self.criterionGAN(pred_fake_bnormal, True)
            self.loss_drm += self.loss_fgan + self.loss_bgan

        self.loss_drm.backward()
    
    def backward_FND(self):
        # Fake; stop backprop to the generator by detaching imbn_pred
        pred_fake = self.netFND(torch.cat([self.im_mask, self.imfn_pred.detach()], 1))
        loss_FND_fake = self.criterionGAN(pred_fake, False)

        # Real
        pred_real = self.netFND(torch.cat([self.im_mask, self.imfn], 1))
        loss_FND_real = self.criterionGAN(pred_real, True)

        # combine loss and calculate gradients
        self.loss_FND = (loss_FND_fake + loss_FND_real) * 0.5
        self.loss_FND.backward()
    
    def backward_BND(self):
        # Fake; stop backprop to the generator by detaching imbn_pred
        pred_fake = self.netBND(torch.cat([self.im_mask, self.imbn_pred.detach()], 1))
        loss_BND_fake = self.criterionGAN(pred_fake, False)

        # Real
        pred_real = self.netBND(torch.cat([self.im_mask, self.imbn], 1))
        loss_BND_real = self.criterionGAN(pred_real, True)

        # combine loss and calculate gradients
        self.loss_BND = (loss_BND_fake + loss_BND_real) * 0.5
        self.loss_BND.backward()


    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward() # compute (fake) output from G

        if self.use_gan_loss: #  update D
            self.set_requires_grad(self.netFND, True)  # enable backprop for FND
            self.optimizer_FND.zero_grad()             # set FND's gradients to zero
            self.backward_FND()                        # calculate gradients for FND
            self.optimizer_FND.step()                  # update FND's weights
            self.set_requires_grad(self.netFND, False) # FND requires no gradients when optimizing G

            self.set_requires_grad(self.netBND, True)  # enable backprop for BND
            self.optimizer_BND.zero_grad()             # set BND's gradients to zero
            self.backward_BND()                        # calculate gradients for BND
            self.optimizer_BND.step()                  # update BND's weights
            self.set_requires_grad(self.netBND, False) # BND requires no gradients when optimizing G
        
        self.optimizer_G.zero_grad() # clear network DRM's existing gradients
        self.backward_G()            # calculate gradients for network DRM
        self.optimizer_G.step()       # update gradients for network DRM
    
    def compute_visuals(self):
        """Calculate additional output images for tensorbard visualization"""
        self.imfd_diff = self.imfd_pred - self.imfd
        self.imbd_diff = self.imbd_pred - self.imbd
        if self.use_grad_loss:
            self.fgrad_pred_x = self.fgrad_pred[:,0,:,:].unsqueeze(1)
            self.fgrad_pred_y = self.fgrad_pred[:,1,:,:].unsqueeze(1)
            self.bgrad_pred_x = self.bgrad_pred[:,0,:,:].unsqueeze(1)
            self.bgrad_pred_y = self.bgrad_pred[:,1,:,:].unsqueeze(1)
            self.fgrad_x = self.fgrad[:,0,:,:].unsqueeze(1)
            self.fgrad_y = self.fgrad[:,1,:,:].unsqueeze(1)
            self.bgrad_x = self.bgrad[:,0,:,:].unsqueeze(1)
            self.bgrad_y = self.bgrad[:,1,:,:].unsqueeze(1)
            self.fgrad_x_diff = self.fgrad_pred_x - self.fgrad_x
            self.fgrad_y_diff = self.fgrad_pred_y - self.fgrad_y
            self.bgrad_x_diff = self.bgrad_pred_x - self.bgrad_x
            self.bgrad_y_diff = self.bgrad_pred_y - self.bgrad_y
        if self.use_normal_loss:
            self.imfn_diff = -torch.nn.functional.cosine_similarity(self.imfn_pred, self.imfn, dim=1, eps=1e-12).unsqueeze(1)
            self.imbn_diff = -torch.nn.functional.cosine_similarity(self.imbn_pred, self.imbn, dim=1, eps=1e-12).unsqueeze(1)