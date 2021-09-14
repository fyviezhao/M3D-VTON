import torch
from torch.utils.tensorboard import SummaryWriter
import os
import time
from . import util


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses tensorboard for display.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a tensorboard server
        Step 3:  create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.use_tensorboard = not opt.no_tensorboard
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.saved = False
        self.ncols = opt.display_ncols


        if self.use_tensorboard: # create board
            tensorboard_dir = os.path.join(opt.checkpoints_dir, opt.datamode, opt.name, 'tensorboard')
            print('create tensorboard directory %s...' % tensorboard_dir)
            util.mkdir(tensorboard_dir)
            self.board = SummaryWriter(log_dir = tensorboard_dir)

        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.datamode, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def display_current_results(self, visuals, epoch, total_iters):
        """Display current results in tensorboard;

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            total_iters (int) -- current total iterations
        """
        if self.use_tensorboard:# display images in tensorboard
            img_tensors = []
            img_tensors_list = []
            for visual_tensors in visuals.values():
                if len(img_tensors) < self.ncols:
                    img_tensors.append(visual_tensors)
                else:
                    img_tensors_list.append(img_tensors)
                    img_tensors = []
                    img_tensors.append(visual_tensors)
            img_tensors_list.append(img_tensors)

            self.board_add_images(self.board, 'Visuals', img_tensors_list, total_iters)

    def tensor_for_board(self, img_tensor):
        # map into [0,1]
        tensor = (img_tensor.clone()+1) * 0.5
        tensor.cpu().clamp(0,1)

        if tensor.size(1) == 1:
            tensor = tensor.repeat(1,3,1,1)

        return tensor

    def tensor_list_for_board(self, img_tensors_list):
        grid_h = len(img_tensors_list)
        grid_w = max(len(img_tensors)  for img_tensors in img_tensors_list)

        batch_size, channel, height, width = self.tensor_for_board(img_tensors_list[0][0]).size()
        canvas_h = grid_h * height
        canvas_w = grid_w * width
        canvas = torch.FloatTensor(batch_size, channel, canvas_h, canvas_w).fill_(0.5)
        for i, img_tensors in enumerate(img_tensors_list):
            for j, img_tensor in enumerate(img_tensors):
                offset_h = i * height
                offset_w = j * width
                tensor = self.tensor_for_board(img_tensor)
                canvas[:, :, offset_h : offset_h + height, offset_w : offset_w + width].copy_(tensor)

        return canvas
    
    def board_add_image(self, board, tag_name, img_tensor, step_count):
        tensor = self.tensor_for_board(img_tensor)

        for i, img in enumerate(tensor):
            self.board.add_image('%s/%03d' % (tag_name, i), img, step_count)

    def board_add_images(self, board, tag_name, img_tensors_list, step_count):
        tensor = self.tensor_list_for_board(img_tensors_list)

        for i, img in enumerate(tensor):
            self.board.add_image('%s/%03d' % (tag_name, i), img, step_count)

    def plot_current_losses(self, total_iters, losses):
        """display current losses in tensorboard
        Parameters:
            total_iters (int)     -- current total iterations
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
            """
        if self.use_tensorboard:
            for loss_name, loss_value in losses.items():
                self.board.add_scalar('Loss/'+loss_name, loss_value, total_iters)
            # add gpu usage info
            # self.board.add_scalar('gpu allocated', round(torch.cuda.memory_allocated(0)/1024**3,1), total_iters)
            # self.board.add_scalar('gpu reserved', round(torch.cuda.memory_reserved(0)/1024**3,1), total_iters)
        else:
            print('Plot failed, you need set opt.no_tensorboard to False to plot current losses in tensorboard.')

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message