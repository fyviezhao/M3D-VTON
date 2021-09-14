"""Model class.

You can specify '--model TFM' to use this model.
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
sys.path.append("..")
from util import util


class TFMModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.set_defaults(norm='batch')  # You can rewrite default values for this model.
        parser.add_argument('--warproot', type=str, default='results/aligned/MTM/test_pairs', help='path to MTM warping result folder')
        parser.add_argument('--input_segmt', action='store_true', default=True, help='if specified, add segmt to input')
        parser.add_argument('--input_depth', action='store_true', default=True, help='if specified, add initial front depth to input')
        parser.add_argument('--input_nc', type=int, default=7, help='input nc for TFM generator. [base: 7| input segmt: +1 | input depth: +1]')
        parser.add_argument('--output_nc', type=int, default=4, help='output nc for TFM generator')
        parser.add_argument('--num_downs', type=int, default=6, help='the number of downsamplings in TOM generator')
        # parser.set_defaults(netD='n_layers') # [VITION-GAN] use 6 layers discriminator
        # parser.set_defaults(n_layers_D=6) # [VITION-GAN] use 6 layers discriminator
        parser.add_argument('--input_nc_D', type=int, default=12, help='input nc for try-on discriminator')
        parser.add_argument('--add_gan_loss', action='store_true', help='if specified, use gan loss')
        parser.add_argument('--lambda_gan', type=float, default=1.0, help='weight for the gan loss')
        parser.add_argument('--lambda_l1', type=float, default=1.0, help='weight for the l1 loss')
        parser.add_argument('--lambda_vgg', type=float, default=1.0, help='weight for the vgg loss')
        parser.add_argument('--lambda_mask', type=float, default=1.0, help='weight for the mask loss')
        
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
        self.input_segmt = opt.input_segmt
        self.input_depth = opt.input_depth
        self.use_gan_loss = opt.add_gan_loss
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['G', 'l1', 'vgg', 'mask']
        if self.use_gan_loss:
            self.loss_names.extend(['gan','D'])
        
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['im_hhl','segmt_vis','c','cm','m_composite', 'pcm', 'p_rendered','p_tryon','im']
        
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        if self.isTrain and self.use_gan_loss:
            self.model_names = ['TFM', 'D']
        else:
            self.model_names = ['TFM']
        

        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        if self.input_segmt:
            opt.input_nc += 1
        if self.input_depth:
            opt.input_nc += 1
        self.netTFM = networks.define_TFM(opt.input_nc, opt.output_nc, opt.num_downs, opt.ngf, opt.norm, opt.use_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            self.netD = networks.define_D(opt.input_nc_D, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # only defined during training time
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionVGG = networks.VGGLoss(device=self.device)
            self.criterionMask = torch.nn.L1Loss()
            if self.use_gan_loss:
                self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            
            # define and initialize optimizers. You can define one optimizer for each network.
            self.optimizer_G = torch.optim.Adam(self.netTFM.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizers = [self.optimizer_G]
            if self.use_gan_loss:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
                self.optimizers.append(self.optimizer_D)

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.im_name = input['im_name']                        # meta info
        self.c_name = input['c_name']                          # meta info
        self.segmt = input['person_parse'].to(self.device)              # for input          
        self.imfd_initial = input['initial_fdepth'].to(self.device)     # for input
        self.c = input['cloth'].to(self.device)                # for input
        self.cm = input['cloth_mask'].to(self.device)          # for input
        self.im_hhl = input['head_hand_lower'].to(self.device) # for input
        self.im = input['person'].to(self.device)              # for ground truth
        self.pcm = input['parse_cloth_mask'].to(self.device)   # for ground truth

       
    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        if self.input_depth and self.input_segmt:
            self.input = torch.cat([self.c,self.cm, self.im_hhl, self.segmt, self.imfd_initial],1)
        elif self.input_segmt:
            self.input = torch.cat([self.c,self.cm, self.im_hhl, self.segmt],1)
        elif self.input_depth:
            self.input = torch.cat([self.c,self.cm, self.im_hhl, self.imfd_initial],1)
        else:
            self.input = torch.cat([self.c,self.cm, self.im_hhl],1)
        outputs = self.netTFM(self.input)
        self.p_rendered, self.m_composite= torch.split(outputs, [3,1], 1)
        self.p_rendered = torch.tanh(self.p_rendered)
        self.m_composite = torch.sigmoid(self.m_composite)
        self.p_tryon = self.c * self.m_composite + self.p_rendered * (1 - self.m_composite)

    def backward_G(self):
        """Calculate losses, gradients; called in every training iteration"""
        # losses for generator only
        self.loss_l1 = self.opt.lambda_l1 * self.criterionL1(self.p_tryon, self.im)
        self.loss_vgg = self.opt.lambda_vgg * self.criterionVGG(self.p_tryon, self.im)
        self.loss_mask = self.opt.lambda_mask * self.criterionMask(self.m_composite, self.pcm)

        if self.use_gan_loss: # G(fake_input) should fake the discriminator
            pred_fake_tryon = self.netD(torch.cat([self.input,self.p_tryon], 1))
            self.loss_gan = self.opt.lambda_gan * self.criterionGAN(pred_fake_tryon, True)

        # combine loss and calculate gradients
        self.loss_G = self.loss_l1 + self.loss_vgg + self.loss_mask
        if self.use_gan_loss:
            self.loss_G = self.loss_G + self.loss_gan #/ self.batch_size 
        
        self.loss_G.backward()
    
    def backward_D(self):
        # Fake; stop backprop to the generator by detaching p_tryon
        pred_fake = self.netD(torch.cat([self.input,self.p_tryon.detach()], 1))
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        pred_real = self.netD(torch.cat([self.input,self.im], 1))
        loss_D_real = self.criterionGAN(pred_real, True)

        # combine loss and calculate gradients
        self.loss_D = (loss_D_fake + loss_D_real) * 0.5
        self.loss_D.backward()


    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward() # compute (fake) output from G

        if self.use_gan_loss: #  update D
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()             # set D's gradients to zero
            self.backward_D()                        # calculate gradients for D
            self.optimizer_D.step()                  # update D's weights
            self.set_requires_grad(self.netD, False) # D requires no gradients when optimizing G
        
        self.optimizer_G.zero_grad() # clear network TFM's existing gradients
        self.backward_G()            # calculate gradients for network TOM
        self.optimizer_G.step()       # update gradients for network TOM
    
    def compute_visuals(self):
        """Calculate additional output images for tensorbard visualization"""
        # convert raw scores to 1-channel mask that can be visualized
        self.segmt_vis = util.decode_labels(self.segmt.int())