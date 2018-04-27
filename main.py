import os
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from CustomDatasetLoader import CustomDataset
from CoreModels.repository import repository
from helper import Tools
from logger import Logger
from PIL import Image
from train_dcgan import Train_DCGAN
from train_wgan import Train_WGAN
from train_infogan import Train_INFOGAN
import yaml


class Model(object):
    def __init__(self, config):
        self.config = config 
        
    # trains the gan based on the type: dcgan, wgan, infogan
    def train(self):        
        if self.config['gan_type'] == 'dcgan':
            Train_DCGAN(self.config).train()
        elif self.config['gan_type'] == 'wgan':
            Train_WGAN(self.config).train()
        elif self.config['gan_type'] == 'infogan':
            Train_INFOGAN(self.config).train()

    # predicts the result
    def predict(self):
        #a batch of samples are loaded from dataloader 
        for sample in self.data_loader:            
            CorrectImages = sample['CorrectImages']
            CorrectEmbedding = sample['CorrectEmbedding']
            txt = sample['txt']

            #if path doesnot exist, path is made
            if not os.path.exists('results/{0}'.format(self.save_directory_path)):
                os.makedirs('results/{0}'.format(self.save_directory_path))

            CorrectImages = Variable(CorrectImages.float()).cuda()
            CorrectEmbedding = Variable(CorrectEmbedding.float()).cuda()
            
            # add random noise to generate multiple images 
            noise = Variable(torch.randn(CorrectImages.size(0), 100)).cuda()
            noise = noise.view(noise.size(0), 100, 1, 1)
            Unrealimages = self.generator(CorrectEmbedding, noise)

            self.logger.draw(CorrectImages, Unrealimages)

            #results are saved
            for image, t in zip(Unrealimages, txt):
                im = Image.fromarray(image.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
                im.save('results/{0}/{1}.jpg'.format(self.save_directory_path, t.replace("/", "")[:100]))
                print(t)

if __name__=='__main__':
    config = None 
    with open('config.yaml', 'r') as myfile:
        config = yaml.load(myfile)
    
    # print("\nCONFIGURATION:")
    # print(config)

    if config:
        model = Model(config)
        if not config['inference']:
            model.train()
        else:
            model.predict()