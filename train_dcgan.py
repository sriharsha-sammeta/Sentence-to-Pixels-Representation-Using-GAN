import os
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from CustomDatasetLoader import CustomDataset
from helper import Tools
from logger import Logger
from PIL import Image
from train_gan import Train_GAN
import yaml

#ys3055
class Train_DCGAN(Train_GAN):
    """ DCGAN training classs """ 
    def __init__(self, config):
        super().__init__(config)
            

    def train(self):
        """ Train method for DCGAN """ 
        l1_penality, l2_penality, metric, = nn.L1Loss(), nn.MSELoss(), nn.BCELoss()        

        for epoch in range(self.num_epochs):
            for sample in self.data_loader:                

                CorrectImages, CorrectEmbedding, IncorrectImages = \
                Variable(sample['CorrectImages'].float()).cuda(), \
                Variable(sample['CorrectEmbedding'].float()).cuda(), \
                Variable(sample['IncorrectImages'].float()).cuda()

                original_keys = torch.ones(CorrectImages.size(0))
                Unreallabels = torch.zeros(CorrectImages.size(0))

                #One sided label smoothing: to prevent discriminator from overpowering
                smoothed_original_keys = torch.FloatTensor(Tools.smooth_label(original_keys.numpy(), -0.1))

                original_keys = Variable(original_keys).cuda()
                smoothed_original_keys = Variable(smoothed_original_keys).cuda()
                Unreallabels = Variable(Unreallabels).cuda()

                # Train the discriminator
                self.discriminator.zero_grad()
                outputs, val_act_original = self.discriminator(CorrectImages, CorrectEmbedding)
                real_loss = metric(outputs, smoothed_original_keys)
                real_score = outputs
                
                noise = Variable(torch.randn(CorrectImages.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                Unrealimages = self.generator(CorrectEmbedding, noise)
                outputs, _ = self.discriminator(Unrealimages, CorrectEmbedding)
                Unrealloss = metric(outputs, Unreallabels)
                Unrealscore = outputs

                d_loss = real_loss + Unrealloss

                d_loss.backward()
                self.optimD.step()

                # Train the generator
                self.generator.zero_grad()
                noise = Variable(torch.randn(CorrectImages.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                Unrealimages = self.generator(CorrectEmbedding, noise)
                outputs, val_act_unreal = self.discriminator(Unrealimages, CorrectEmbedding)
                _, val_act_original = self.discriminator(CorrectImages, CorrectEmbedding)

                val_act_unreal = torch.mean(val_act_unreal, 0)
                val_act_original = torch.mean(val_act_original, 0)


                # Loss function                
                g_loss = metric(outputs, original_keys) \
                         + self.l2_val * l2_penality(val_act_unreal, val_act_original.detach()) \
                         + self.l1_val * l1_penality(Unrealimages, CorrectImages)

                g_loss.backward()
                self.optimG.step()

                self.logger.logger_iter_gan(epoch,d_loss, g_loss, real_score, Unrealscore)
                self.logger.draw(CorrectImages, Unrealimages)

            self.logger.plotter_epoch_w(epoch)

            if (epoch) % 10 == 0:
                Tools.save_checkpoint(self.discriminator, self.generator, self.path_to_checkpoints, self.save_directory_path, epoch)
