import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler
from torchvision import models
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
import functools
import random
import sys
sys.path.append("..")
from util import util


###############################################################################
# Helper Functions
###############################################################################

class Vgg19(nn.Module):
    """ Vgg19 for VGGLoss. """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class Sobel(nn.Module):
    """ Soebl operator to calculate depth grad. """

    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """x: depth map (batch_size,1,H,W)"""
        out = self.edge_conv(x) 
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
  
        return out

class FeatureExtraction(nn.Module):
    """ 
    size: 512-256-128-64-32-32-32
    channel: in_nc-64-128-256-512-512-512
    """
    def __init__(self, input_nc, ngf=64, n_layers=3, norm_layer=nn.InstanceNorm2d,  use_dropout=False):
        super(FeatureExtraction, self).__init__()
        downconv = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        model = [downconv, nn.ReLU(True), norm_layer(ngf)]
        for i in range(n_layers):
            in_ngf = 2**i * ngf if 2**i * ngf < 512 else 512
            out_ngf = 2**(i+1) * ngf if 2**i * ngf < 512 else 512
            downconv = nn.Conv2d(in_ngf, out_ngf, kernel_size=4, stride=2, padding=1)
            model += [downconv, nn.ReLU(True)]
            model += [norm_layer(out_ngf)]
        model += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(True)]
        model += [norm_layer(512)]
        model += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(True)]

        self.model = nn.Sequential(*model)
        # init_weights(self.model, init_type='normal

    def forward(self, x):
        return self.model(x)

class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature,norm)

class FeatureCorrelation(nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()
    
    def forward(self, feature_A, feature_B):
        b,c,h,w = feature_A.size()
        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
        feature_B = feature_B.view(b,c,h*w).transpose(1,2)
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B,feature_A)
        correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
        return correlation_tensor
    
class FeatureRegression(nn.Module):
    def __init__(self, input_nc=640, output_dim=6):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(64 * 8 * 5, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x) # (batch_size,64,8,5)
        x = x.reshape(x.size(0), -1) # (batch_size,2560)
        x = self.linear(x) # (batch_size,output_dim)
        x = self.tanh(x)
        return x

class AffineGridGen(nn.Module):
    def __init__(self, out_h=512, out_w=320, out_ch = 3):
        super(AffineGridGen, self).__init__()        
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch
        
    def forward(self, theta):
        bs=theta.size()[0]
        if not theta.size()==(bs,2,3):
            theta = theta.view(-1,2,3)
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size,self.out_ch,self.out_h,self.out_w))
        return F.affine_grid(theta, out_size)

class TpsGridGen(nn.Module):
    def __init__(self, out_h=512, out_w=320, use_regular_grid=True, grid_size=5, device='cpu'):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w

        # create grid in numpy
        self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32) # (512,320,3)
        # sampling grid using meshgrid
        self.grid_X,self.grid_Y = np.meshgrid(np.linspace(-1,1,out_w),np.linspace(-1,1,out_h))
        # grid_X,grid_Y: size [1,H,W,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3) # (1,512,320,1)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3) # (1,512,320,1)
        if device != 'cpu':
            self.grid_X = self.grid_X.to(device)
            self.grid_Y = self.grid_Y.to(device)

        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = np.linspace(-1,1,grid_size)
            self.N = grid_size*grid_size # 25 control points
            P_Y, P_X = np.meshgrid(axis_coords,axis_coords) # BUG: should return (P_X, P_Y)?
            # P_X, P_Y = np.meshgrid(axis_coords,axis_coords)
            P_X = np.reshape(P_X,(-1,1)) # size (N=25,1)
            P_Y = np.reshape(P_Y,(-1,1)) # size (N=25,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.P_X_base = P_X.clone() # size (N=25,1)
            self.P_Y_base = P_Y.clone() # size (N=25,1)
            self.Li = self.compute_L_inverse(P_X,P_Y).unsqueeze(0) # (1,N+3=28,N+3=28)
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4) # (1,1,1,1,N=25)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4) # (1,1,1,1,N=25)
            if device != 'cpu':
                self.P_X = self.P_X.to(device)
                self.P_Y = self.P_Y.to(device)
                self.P_X_base = self.P_X_base.to(device)
                self.P_Y_base = self.P_Y_base.to(device)
                self.Li = self.Li.to(device)

            
    def forward(self, theta):
        # theta.size(): (batch_size, N*2=50)
        warped_grid = self.apply_transformation(theta,torch.cat((self.grid_X,self.grid_Y),3)) # (batch_size,512,512,2)
        
        return warped_grid
    
    def compute_L_inverse(self,X,Y):
        N = X.size()[0] # num of points (along dim 0)
        # construct matrix K
        Xmat = X.expand(N,N)
        Ymat = Y.expand(N,N)
        # a quick way to calculate distances between every control point pairs
        P_dist_squared = torch.pow(Xmat-Xmat.transpose(0,1),2)+torch.pow(Ymat-Ymat.transpose(0,1),2)
        P_dist_squared[P_dist_squared==0]=1 # make diagonal 1 to avoid NaN in log computation
        # the TPS kernel funciont $U(r) = r^2*log(r)$
        # K.size: (N,N)
        K = torch.mul(P_dist_squared,torch.log(P_dist_squared)) # BUG: should be torch.log(torch.sqrt(P_dist_squared))?
        # construct matrix L
        Z = torch.FloatTensor(N,1).fill_(1)
        O = torch.FloatTensor(3,3).fill_(0)       
        P = torch.cat((Z,X,Y),1) # (N,3)
        L = torch.cat((torch.cat((K,P),1),torch.cat((P.transpose(0,1),O),1)),0) # (N+3,N+3)
        Li = torch.inverse(L) # (N+3,N+3)

        return Li
        
    def apply_transformation(self,theta,points):
        if theta.dim()==2:
            theta = theta.unsqueeze(2).unsqueeze(3) # (batch_size, N*2=50, 1, 1)
        batch_size = theta.size()[0]
        # input are the corresponding control points P_i
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords  
        # and points[:,:,:,1] are the Y coords.  
        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]
        
        # split theta into point coordinates (extract the displacements Q_X and Q_Y from theta)
        Q_X=theta[:,:self.N,:,:].squeeze(3) # (batch_size, N=25, 1)
        Q_Y=theta[:,self.N:,:,:].squeeze(3) # (batch_size, N=25, 1)
        # add the displacements to the original control points to get the target control points
        Q_X = Q_X + self.P_X_base.expand_as(Q_X)
        Q_Y = Q_Y + self.P_Y_base.expand_as(Q_Y)

        # compute weigths for non-linear part (multiply by the inverse matrix Li to get the coefficient vector W_X and W_Y)
        W_X = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_X) # (batch_size, N=25, 1)
        W_Y = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_Y) # (batch_size, N=25, 1)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        # compute weights for affine part (calculate the linear part $a_1 + a_x*a + a_y*y$)
        A_X = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_X) # (batch_size, 3, 1)
        A_Y = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_Y) # (batch_size, 3, 1)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1) 
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        
        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = self.P_X.expand((1,points_h,points_w,1,self.N)) # (1,512,320,1,N=25)
        P_Y = self.P_Y.expand((1,points_h,points_w,1,self.N)) # (1,512,320,1,N=25)

        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        # points: size [1,H,W,2]
        # points_X_for_summation, points_Y_for_summation: size [1,H,W,1,N]
        points_X_for_summation = points[:,:,:,0].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,0].size()+(1,self.N))
        points_Y_for_summation = points[:,:,:,1].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,1].size()+(1,self.N))
        
        if points_b==1:
            delta_X = points_X_for_summation-P_X # (1,512,320,1,N=25)
            delta_Y = points_Y_for_summation-P_Y # (1,512,320,1,N=25)
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation-P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation-P_Y.expand_as(points_Y_for_summation)
            
        dist_squared = torch.pow(delta_X,2)+torch.pow(delta_Y,2)  # (1,512,320,1,N=25)
        dist_squared[dist_squared==0]=1 # avoid NaN in log computation
        # pass the distances to the radial basis function U
        # U: size [1,H,W,1,N]
        U = torch.mul(dist_squared,torch.log(dist_squared)) 
        
        # expand grid in batch dimension if necessary
        points_X_batch = points[:,:,:,0].unsqueeze(3) # (1,512,320,1)
        points_Y_batch = points[:,:,:,1].unsqueeze(3) # (1,512,320,1)
        if points_b==1:
            points_X_batch = points_X_batch.expand((batch_size,)+points_X_batch.size()[1:]) # (batch_size,512,320,1)
            points_Y_batch = points_Y_batch.expand((batch_size,)+points_Y_batch.size()[1:]) # (batch_size,512,320,1)
        
        # points_X_prime, points_Y_prime: size [B,H,W,1]
        points_X_prime = A_X[:,:,:,:,0]+ \
                       torch.mul(A_X[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_X[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_X,U.expand_as(W_X)),4)
                    
        points_Y_prime = A_Y[:,:,:,:,0]+ \
                       torch.mul(A_Y[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_Y[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_Y,U.expand_as(W_Y)),4)
        
        # concatenate dense array points points_X_prime and points_Y_prime into a grid
        return torch.cat((points_X_prime,points_Y_prime),3)

class DepthDec(nn.Module):
    """
    size: 32-32-32-64-128-256-512
    channel: in_nc-512-512-256-128-64-out_nc
    """
    def __init__(self, in_nc=1024, out_nc=2):
        super(DepthDec, self).__init__()
        self.upconv52 = nn.Conv2d(in_nc, 512, kernel_size=3, stride=1, padding=1)
        self.upnorm52 = nn.InstanceNorm2d(512)
        self.upconv51 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.upnorm51 = nn.InstanceNorm2d(512)

        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upconv4 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.upnorm4 = nn.InstanceNorm2d(256)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upconv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.upnorm3 = nn.InstanceNorm2d(128)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upconv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.upnorm2 = nn.InstanceNorm2d(64)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upconv1 = nn.Conv2d(64, out_nc, kernel_size=3, stride=1, padding=1)
        self.upnorm1 = nn.InstanceNorm2d(out_nc)

    def forward(self, x):
        x52up = F.relu_(self.upnorm52(self.upconv52(x)))	        
        x51up = F.relu_(self.upnorm51(self.upconv51(x52up)))	        
        x4up = F.relu_(self.upnorm4(self.upconv4(self.upsample4(x51up))))	        
        x3up = F.relu_(self.upnorm3(self.upconv3(self.upsample3(x4up))))	        
        x2up = F.relu_(self.upnorm2(self.upconv2(self.upsample2(x3up))))	        
        x1up = self.upnorm1(self.upconv1(self.upsample1(x2up)))	        

        return x1up

class SegmtDec(nn.Module):
    """
    size: 32-32-32-64-128-256-512
    channel: in_nc-512-512-256-128-64-out_nc
    """
    def __init__(self, in_nc=1024, out_nc=20):
        super(SegmtDec, self).__init__()
        self.upconv52 = nn.Conv2d(in_nc, 512, kernel_size=3, stride=1, padding=1)
        self.upnorm52 = nn.InstanceNorm2d(512)
        self.upconv51 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.upnorm51 = nn.InstanceNorm2d(512)

        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upconv4 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.upnorm4 = nn.InstanceNorm2d(256)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upconv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.upnorm3 = nn.InstanceNorm2d(128)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upconv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.upnorm2 = nn.InstanceNorm2d(64)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upconv1 = nn.Conv2d(64, out_nc, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x52up = F.relu_(self.upnorm52(self.upconv52(x)))	        
        x51up = F.relu_(self.upnorm51(self.upconv51(x52up)))	        
        x4up = F.relu_(self.upnorm4(self.upconv4(self.upsample4(x51up))))	        
        x3up = F.relu_(self.upnorm3(self.upconv3(self.upsample3(x4up))))	        
        x2up = F.relu_(self.upnorm2(self.upconv2(self.upsample2(x3up))))	        
        x1up = self.upconv1(self.upsample1(x2up))	        

        return x1up

class UnetSkipConnectionBlock(nn.Module):
    """Defines the submodule with skip connection.
    X -------------------identity---------------------- X
      |-- downsampling -- |submodule| -- upsampling --|
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv]
            up = [uprelu, upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upsample, upconv, upnorm]
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            upconv = nn.Conv2d(inner_nc*2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsample, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = self._l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = self._l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _l2normalize(self, x, eps=1e-12):
        return x / (x.norm() + eps)

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = self._l2normalize(u.data)
        v.data = self._l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return torch.nn.Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs_keep> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs_keep) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs_keep, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

def random_crop(reals, fakes, winsize=48):
    y, x = [random.randint(reals.size(i)//4, int(reals.size(i)*0.75)-winsize-1) for i in (2, 3)]
    return reals[:,:,y:y+winsize,x:x+winsize], fakes[:,:,y:y+winsize,x:x+winsize]

def define_MTM(input_nc_A=29, input_nc_B=3, ngf=64, n_layers=3, img_height=512, img_width=320, grid_size=5, add_tps=True, 
            add_depth=True, add_segmt=True, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create MTM model. 

    Parameters:
        input_nc_A (int)   -- the number of channels of agnostic input (default: 11)
        input_nc_B (int)   -- the number of channels of flat cloth mask input (default: 3)
        ngf (int)          -- the number of filters in the first conv layer (default: 64)
        img_height (int)   -- input image height (default: 512)
        img_width (int)    -- input image width (default: 320)
        norm (str)         -- the name of normalization layers used in the network: batch | instance | none (default: instance)
        use_dropout (bool) -- whether to use dropout in feature extraction module (default: False)
        init_type (str)    -- the name of our initialization method (default: normal)
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal (default: 0.02)
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2 (default: [])

    Returns:
        a generator, the generator has been initialized by <init_net>.
    """
    
    norm_layer = get_norm_layer(norm_type=norm)
    device = f'cuda:{gpu_ids[0]}' if len(gpu_ids) > 0 else 'cpu'
    net = MTM(input_nc_A, input_nc_B, ngf, n_layers, img_height, img_width, grid_size, add_tps, add_depth, add_segmt, norm_layer, use_dropout, device)
    
    return init_net(net, init_type, init_gain, gpu_ids)

def define_TFM(input_nc=9, output_nc=4, num_downs=6, ngf=64, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)

    net = UnetGenerator(input_nc, output_nc, num_downs, ngf, norm_layer, use_dropout)

    return init_net(net, init_type, init_gain, gpu_ids)

def define_DRM(input_nc=4, output_nc=2, ngf=32, norm='instanc', init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)

    net = DRM(input_nc, output_nc, ngf, norm_layer)

    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

        [spectral_norm]: DCGAN-like spectral norm discriminator based on the SNGAN

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'spectral_norm':
        net = SNDiscriminator(input_nc, ndf)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)

##############################################################################
# Generators and Discriminators
##############################################################################
class MTM(nn.Module):
    def __init__(self, input_nc_A=29, input_nc_B=3, ngf=64, n_layers=3, img_height=512, img_width=320, grid_size=5, 
                add_tps=True, add_depth=True, add_segmt=True, norm_layer=nn.InstanceNorm2d, use_dropout=False, device='cpu'):
        super(MTM, self).__init__()
        self.add_tps = add_tps
        self.add_depth = add_depth
        self.add_segmt = add_segmt

        self.extractionA = FeatureExtraction(input_nc_A, ngf, n_layers, norm_layer, use_dropout)
        self.extractionB = FeatureExtraction(input_nc_B, ngf, n_layers, norm_layer, use_dropout)
        self.l2norm = FeatureL2Norm()
        self.correlation = FeatureCorrelation()
        self.regression_tps = FeatureRegression(input_nc=640, output_dim=2*grid_size**2)
        self.tps_grid_gen = TpsGridGen(img_height, img_width, grid_size=grid_size, device=device)

        if self.add_segmt:
            self.segmt_dec = SegmtDec()

        if self.add_depth:
            self.depth_dec = DepthDec(in_nc=1024)

    def forward(self, inputA, inputB):
        """ 
            input A: agnostic (batch_size,12,512,320)
            input B: flat cloth mask(batch_size,1,512,320)
        """
        output = {'theta_tps':None, 'grid_tps':None, 'depth':None, 'segmt':None}
        featureA = self.extractionA(inputA) # featureA: size (batch_size,512,32,20)
        featureB = self.extractionB(inputB) # featureB: size (batch_size,512,32,20)
        if self.add_depth or self.add_segmt:
            featureAB = torch.cat([featureA, featureB], 1) # input for DepthDec and SegmtDec: (batch_size,1024,32,20)
            if self.add_depth:
                depth_pred = self.depth_dec(featureAB)
                output['depth'] = depth_pred
            if self.add_segmt:
                segmt_pred = self.segmt_dec(featureAB)
                output['segmt'] = segmt_pred
        if self.add_tps:
            featureA = self.l2norm(featureA)
            featureB = self.l2norm(featureB)
            correlationAB = self.correlation(featureA, featureB) # correlationAB: size (batch_size, 640, 32, 32)
            theta_tps = self.regression_tps(correlationAB)
            grid_tps = self.tps_grid_gen(theta_tps)
            output['theta_tps'], output['grid_tps'] = theta_tps, grid_tps

        return output

class UnetGenerator(nn.Module):
    """Defines the Unet generator."""
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True) # add the innermost layer
        for i in range(num_downs - 5): # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class DRM(nn.Module):
    def __init__(self, in_channel, out_channel, ngf=32, norm_layer=nn.InstanceNorm2d):
        super(DRM, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.ngf = ngf
        
        # size -> size / 2
        self.l0 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.ngf, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.ngf * 2, 3, padding=1, stride=2),
            nn.ELU(),
            norm_layer(self.ngf * 2)
        )

        # size / 2 -> size / 4
        self.l1 = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 3, padding=1, stride=2),
            nn.ELU(),
            norm_layer(self.ngf * 4)
        )

        # size / 4 -> size / 8
        self.l2 = nn.Sequential(
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 8, 3, padding=1, stride=2),
            nn.ELU(),
            norm_layer(self.ngf * 8)
        )

        # size / 8 -> size / 16
        self.l3 = nn.Sequential(
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 16, 3, padding=1, stride=2),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            norm_layer(self.ngf * 16)
        )

        self.block1 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        self.block2 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        # size / 16 -> size / 8
        self.l3u = nn.Sequential(
            nn.Conv2d(self.ngf * 24, self.ngf * 8, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            norm_layer(self.ngf * 8)
        )

        # size / 8 -> size / 4
        self.l2u = nn.Sequential(
            nn.Conv2d(self.ngf * 12, self.ngf * 4, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            norm_layer(self.ngf * 4)
        )

        # size / 4 -> size / 2
        self.l1u = nn.Sequential(
            nn.Conv2d(self.ngf * 6, self.ngf * 2, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            norm_layer(self.ngf * 2)
        )

        # size / 2 -> size
        self.l0u = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.ngf, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.out_channel, 3, padding=1, stride=1),
            nn.Tanh()
        )

    def forward(self, input_data, inter_mode='bilinear'):
        x0 = self.l0(input_data)
        x1 = self.l1(x0)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x3 = self.block1(x3) + x3
        x3 = self.block2(x3) + x3
        x3u = nn.functional.interpolate(x3, size=x2.shape[2:4], mode=inter_mode)
        x3u = self.l3u(torch.cat((x3u, x2), dim=1))
        x2u = nn.functional.interpolate(x3u, size=x1.shape[2:4], mode=inter_mode)
        x2u = self.l2u(torch.cat((x2u, x1), dim=1))
        x1u = nn.functional.interpolate(x2u, size=x0.shape[2:4], mode=inter_mode)
        x1u = self.l1u(torch.cat((x1u, x0), dim=1))
        x0u = nn.functional.interpolate(x1u, size=input_data.shape[2:4], mode=inter_mode)
        x0u = self.l0u(x0u)
        return x0u


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

class SNDiscriminator(nn.Module):
    """Defines a DCGAN-like spectral norm discriminator (SNGAN)"""
    def __init__(self, input_nc, ndf=64):
        super(SNDiscriminator, self).__init__()

        self.conv1 = SpectralNorm(nn.Conv2d(input_nc, ndf, 3, stride=1, padding=(1,1)))

        self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1,1)))
        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(1,1)))
        self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1,1)))
        self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, padding=(1,1)))
        self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))
        self.conv7 = SpectralNorm(nn.Conv2d(256, 512, 3, stride=1, padding=(1,1)))


        self.fc = SpectralNorm(nn.Linear(4 * 4 * 512, 1))

    def forward(self, x):
        m = x
        m = nn.LeakyReLU(0.1)(self.conv1(m))
        m = nn.LeakyReLU(0.1)(self.conv2(m))
        m = nn.LeakyReLU(0.1)(self.conv3(m))
        m = nn.LeakyReLU(0.1)(self.conv4(m))
        m = nn.LeakyReLU(0.1)(self.conv5(m))
        m = nn.LeakyReLU(0.1)(self.conv6(m))
        m = nn.LeakyReLU(0.1)(self.conv7(m))

        return self.fc(m.view(-1, 4 * 4 * 512))

##############################################################################
# Loss Functions
##############################################################################

class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()

    def forward(self, depth_pred, depth_gt):
        loss_depth = torch.log(torch.abs(depth_pred - depth_gt) + 1).mean()
        
        return loss_depth

class DepthGradLoss(nn.Module):
    def __init__(self):
        super(DepthGradLoss, self).__init__()

    def forward(self, depth_grad_pred, depth_grad_gt):
        depth_grad_gt_dx = depth_grad_gt[:, 0, :, :].unsqueeze(1)
        depth_grad_gt_dy = depth_grad_gt[:, 1, :, :].unsqueeze(1)
        depth_grad_pred_dx = depth_grad_pred[:, 0, :, :].unsqueeze(1)
        depth_grad_pred_dy = depth_grad_pred[:, 1, :, :].unsqueeze(1)
        
        loss_dx = torch.log(torch.abs(depth_grad_pred_dx - depth_grad_gt_dx) + 1).mean()
        loss_dy = torch.log(torch.abs(depth_grad_pred_dy - depth_grad_gt_dy) + 1).mean()
        
        loss_grad = loss_dx + loss_dy
    
        return loss_grad

class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-12)

    def forward(self, normal_pred, normal_gt):
        
        loss_normal = (1 - self.cos(normal_pred, normal_gt)).mean()
        
        return loss_normal

class WeakInlierCount(nn.Module):
    def __init__(self, tps_grid_size=5, h_matches=15, w_matches=15, dilation_filter=0, device=[]):
        super(WeakInlierCount, self).__init__()
        self.normalize=normalize_inlier_count

        self.geometricTnf = TpsGridGen(out_h=h_matches, out_w=w_matches, use_regular_grid=True, grid_size=tps_grid_size, device=self.device)

        # define identity mask tensor (w,h are switched and will be permuted back later)
        mask_id = np.zeros((w_matches,h_matches,w_matches*h_matches))
        idx_list = list(range(0, mask_id.size, mask_id.shape[2]+1))
        mask_id.reshape((-1))[idx_list]=1
        mask_id = mask_id.swapaxes(0,1)

        # perform 2D dilation to each channel 
        if not (isinstance(dilation_filter,int) and dilation_filter==0):
            for i in range(mask_id.shape[2]):
                mask_id[:,:,i] = binary_dilation(mask_id[:,:,i],structure=dilation_filter).astype(mask_id.dtype)
            
        # convert to PyTorch variable
        self.mask_id = torch.from_numpy(mask_id.transpose(1,2).transpose(0,1).unsqueeze(0).float()).to(self.device)

    def forward(self, theta, matches, return_outliers=False):
        batch_size=theta.size()[0]
        theta=theta.clone()
        mask = self.geometricTnf(util.expand_dim(self.mask_id,0,batch_size),theta)
        mask = self.geometricTnf(self.mask_id.expand())
        if return_outliers:
            mask_outliers = self.geometricTnf(util.expand_dim(1.0-self.mask_id,0,batch_size),theta)

        # normalize inlier conunt
        epsilon=1e-5
        mask = torch.div(mask, torch.sum(torch.sum(torch.sum(mask+epsilon,3),2),1).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(mask))
        if return_outliers:
            mask_outliers = torch.div(mask_outliers, torch.sum(torch.sum(torch.sum(mask_outliers+epsilon,3),2),1).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(mask_outliers))
        
        # compute score
        score = torch.sum(torch.sum(torch.sum(torch.mul(mask,matches),3),2),1)
        if return_outliers:
            score_outliers = torch.sum(torch.sum(torch.sum(torch.mul(mask_outliers,matches),3),2),1)
            return (score,score_outliers)

        return score
    

class ThetaLoss(nn.Module):
    def __init__(self, grid_size=5, device='cpu'):
        super(ThetaLoss, self).__init__()
        self.device = device
        self.grid_size = grid_size
        
    def forward(self, theta):
        batch_size = theta.size()[0]
        coordinate = theta.view(batch_size, -1, 2) # (4,25,2)
        # coordinate+=torch.randn(coordinate.shape).cuda()/10
        row_loss = self.get_row_loss(coordinate, self.grid_size)
        col_loss = self.get_col_loss(coordinate, self.grid_size)
        # row_x, row_y, col_x, col_y: size [batch_size,15]
        row_x, row_y = row_loss[:,:,0], row_loss[:,:,1]
        col_x, col_y = col_loss[:,:,0], col_loss[:,:,1]
        # TODO: what does 0.08 mean?
        if self.device != 'cpu':
            rx, ry, cx, cy = (torch.tensor([0.08]).to(self.device) for i in range(4))
        else:
            rx, ry, cx, cy = (torch.tensor([0.08]) for i in range(4))
        rx_loss = torch.max(rx, row_x).mean()
        ry_loss = torch.max(ry, row_y).mean()
        cx_loss = torch.max(cx, col_x).mean()
        cy_loss = torch.max(cy, col_y).mean()
        sec_diff_loss = rx_loss + ry_loss + cx_loss + cy_loss
        slope_loss = self.get_slope_loss(coordinate, self.grid_size).mean()

        theta_loss = sec_diff_loss + slope_loss

        return theta_loss
    
    def get_row_loss(self, coordinate, num):
        sec_diff = []
        for j in range(num):
            buffer = 0
            for i in range(num-1):
                # TODO: should be L2 distance according to ACGPN paper,  but not L1?
                diff = (coordinate[:, j*num+i+1, :]-coordinate[:, j*num+i, :]) ** 2
                if i >= 1:
                    sec_diff.append(torch.abs(diff-buffer))
                buffer = diff

        return torch.stack(sec_diff, dim=1)
    
    def get_col_loss(self, coordinate, num):
        sec_diff = []
        for i in range(num):
            buffer = 0
            for j in range(num - 1):
                # TODO: should be L2 distance according to ACGPN paper, but not L1?
                diff = (coordinate[:, (j+1)*num+i, :] - coordinate[:, j*num+i, :]) ** 2
                if j >= 1:
                    sec_diff.append(torch.abs(diff-buffer))
                buffer = diff
                
        return torch.stack(sec_diff,dim=1)
    
    def get_slope_loss(self, coordinate, num):
        slope_diff = []
        for j in range(num - 2):
            x, y = coordinate[:, (j+1)*num+1, 0], coordinate[:, (j+1)*num+1, 1]
            x0, y0 = coordinate[:, j*num+1, 0], coordinate[:, j*num+1, 1]
            x1, y1 = coordinate[:, (j+2)*num+1, 0], coordinate[:, (j+2)*num+1, 1]
            x2, y2 = coordinate[:, (j+1)*num, 0], coordinate[:, (j+1)*num, 1]
            x3, y3 = coordinate[:, (j+1)*num+2, 0], coordinate[:, (j+1)*num+2, 1]
            row_diff = torch.abs((y0 - y) * (x1 - x) - (y1 - y) * (x0 - x))
            col_diff = torch.abs((y2 - y) * (x3 - x) - (y3 - y) * (x2 -x))
            slope_diff.append(row_diff + col_diff)
            
        return torch.stack(slope_diff, dim=0)

class GridLoss(nn.Module):
    def __init__(self, image_height, image_width, distance='l1'):
        super(GridLoss, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.distance == distance

    def forward(self, grid):
        gx = grid[:,:,:,0]
        gy = grid[:,:,:,1]
        gx_ctr = gx[:, 1:self.image_height-1, 1:self.image_width-1]
        gx_up = gx[:, 0:self.image_height-2, 1:self.image_width-1]
        gx_down = gx[:, 2:self.image_height, 1:self.image_width-1]
        gx_left = gx[:, 1:self.image_height-1, 0:self.image_width-2]
        gx_right = gx[:, 1:self.image_height-1, 2:self.image_width]

        gy_ctr = gy[:, 1:self.image_height-1, 1:self.image_width-1]
        gy_up = gy[:, 0:self.image_height-2, 1:self.image_width-1]
        gy_down = gy[:, 2:self.image_height, 1:self.image_width-1]
        gy_left = gy[:, 1:self.image_height-1, 0:self.image_width-2]
        gy_right = gy[:, 1:self.image_height-1, 2:self.image_width]

        if self.distance == 'l1':
            grid_loss_left = self._l1_distance(gx_left, gx_ctr)
            grid_loss_right = self._l1_distance(gx_right, gx_ctr)
            grid_loss_up = self._l1_distance(gy_up, gy_ctr)
            grid_loss_down = self._l1_distance(gy_down, gy_ctr)
        elif self.distance == 'l2':
            grid_loss_left = self._l2_distance(gx_left, gy_left, gx_ctr, gy_ctr)
            grid_loss_right = self._l2_distance(gx_right, gy_right, gx_ctr, gy_ctr)
            grid_loss_up = self._l2_distance(gx_up, gy_up, gx_ctr, gy_ctr)
            grid_loss_down = self._l2_distance(gx_down, gy_down, gx_ctr, gy_ctr)

        grid_loss = torch.sum(torch.abs(grid_loss_left-grid_loss_right) + torch.abs(grid_loss_up-grid_loss_down))

        return grid_loss
    
    def _l1_distance(self, x1, x2):

        return torch.abs(x1 - x2)
    
    def _l2_distance(self, x1, y1, x2, y2):
        
        return torch.sqrt(torch.mul(x1-x2, x1-x2) + torch.mul(y1-y2, y1-y2))

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class VGGLoss(nn.Module):
    def __init__(self, layids = None, device = 'cpu'):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        if device != 'cpu':
            self.vgg.to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        
        return loss

class VGGVector(nn.Module):
    def __init__(self, layids = None, device = 'cpu'):
        super(VGGVector, self).__init__()
        self.vgg = Vgg19()
        if device != 'cpu':
            self.vgg.to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        vgg_vector = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            if i == 0:
                vgg_vector += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach()).expand(1,1)
            else:
                vgg_vector = torch.cat([vgg_vector, self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach()).expand(1,1)], 1)

        return vgg_vector
