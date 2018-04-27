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


class Train_INFOGAN(Train_GAN):
    def __init__(self, config):
        super().__init__(config)

    def train(self):     
        one = torch.FloatTensor([1])
        mone = one * -1

        one = Variable(one).cuda()
        mone = Variable(mone).cuda()
        cls = False
        gen_iteration = 0
        for epoch in range(self.num_epochs):
            iterator = 0

            # batch of data taken from dataloader
            data_iterator = iter(self.data_loader)            
            while iterator < len(self.data_loader):                
                # number of iteration counts to avoid overfitting
                if gen_iteration < 25 or gen_iteration % 500 == 0:
                    d_iter_count = 100
                else:
                    d_iter_count = self.d_iterations

                d_iter = 0

                # Training discriminator
                while d_iter < d_iter_count and iterator < len(self.data_loader):
                    d_iter += 1

                    for p in self.discriminator.parameters():
                        p.requires_grad = True

                    self.discriminator.zero_grad()

                    sample = next(data_iterator)
                    iterator += 1

                    #extracting necessary fields
                    CorrectImages = sample['CorrectImages']
                    CorrectEmbedding = sample['CorrectEmbedding']
                    IncorrectImages = sample['IncorrectImages']

                    #converting them to pytorch variables
                    CorrectImages = Variable(CorrectImages.float()).cuda()
                    CorrectEmbedding = Variable(CorrectEmbedding.float()).cuda()
                    IncorrectImages = Variable(IncorrectImages.float()).cuda()

                    #setting up discriminator 
                    outputs, _ = self.discriminator(CorrectImages, CorrectEmbedding)
                    real_loss = torch.mean(outputs)
                    real_loss.backward(mone)

                    #if cls algorithm is enabled, do the following
                    if cls:
                        outputs, _ = self.discriminator(IncorrectImages, CorrectEmbedding)
                        wrong_loss = torch.mean(outputs)
                        wrong_loss.backward(one)

                    # adding random noise
                    noise = Variable(torch.randn(CorrectImages.size(0), self.dimension_noise), volatile=True).cuda()
                    noise = noise.view(noise.size(0), self.dimension_noise, 1, 1)

                    Unrealimages = Variable(self.generator(CorrectEmbedding, noise).data)
                    outputs, _ = self.discriminator(Unrealimages, CorrectEmbedding)
                    infoloss = torch.mean(outputs)
                    infoloss.backward(one)
                    
                    # loss caliculation in discriminator: normal loss + mutual information loss
                    d_loss = real_loss - infoloss

                    if cls:
                        d_loss = d_loss - wrong_loss

                    self.optimD.step()

                    #clamping parameters
                    for p in self.discriminator.parameters():
                        p.data.clamp_(-0.01, 0.01)

                # Training Generator
                for p in self.discriminator.parameters():
                    p.requires_grad = False
                self.generator.zero_grad()
                noise = Variable(torch.randn(CorrectImages.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                Unrealimages = self.generator(CorrectEmbedding, noise)
                outputs, _ = self.discriminator(Unrealimages, CorrectEmbedding)

                # loss caliculation in generator: normal loss + information loss 
                g_loss = torch.mean(outputs)
                g_loss.backward(mone)
                g_loss = - g_loss
                self.optimG.step()

                gen_iteration += 1

                #logging on CLI
                self.logger.draw(CorrectImages, Unrealimages)
                self.logger.logger_iter_wgan(epoch, gen_iteration, d_loss, g_loss, real_loss, infoloss)
                
            self.logger.plotter_epoch(gen_iteration)

            #saving checkpoint
            if (epoch+1) % 50 == 0:
                Tools.save_checkpoint(self.discriminator, self.generator, self.path_to_checkpoints, epoch)           