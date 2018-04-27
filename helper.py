import numpy as np
import os
import pdb
import torch
from torch import nn
from torch import  autograd
from visdom import Visdom
import torchvision
from PIL import ImageDraw, Image, ImageFont

class Tools(object):
    
    @staticmethod
    def randomly_initialize_weights(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    @staticmethod
    def smooth_label(tensor, offset):
        return tensor + offset

    @staticmethod    
    def GP_caliculate(netD, real_data, real_embed, Unrealdata, LAMBDA):
        BATCH_SIZE = real_data.size(0)
        alpha = torch.rand(BATCH_SIZE, 1)
        alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement() / BATCH_SIZE)).contiguous().view(BATCH_SIZE, 3, 64, 64)
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * Unrealdata)

        interpolates = interpolates.cuda()

        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates, _ = netD(interpolates, real_embed)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

        return gradient_penalty

    @staticmethod
    def save_checkpoint(netD, netG, dir_path, subdir_path, epoch):
        path =  os.path.join(dir_path, subdir_path)
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(netD.state_dict(), '{0}/disc_{1}.pth'.format(path, epoch))
        torch.save(netG.state_dict(), '{0}/gen_{1}.pth'.format(path, epoch))



class VisdomPlotter(object):

    """Plots to Visdom"""

    def __init__(self, env_name='gan'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y, xlabel='epoch'):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel=xlabel,
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name)

    def draw(self, var_name, images):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.images(images, env=self.env)
        else:
            self.viz.images(images, env=self.env, win=self.plots[var_name])