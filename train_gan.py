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
import yaml


class Train_GAN(object):
    def __init__(self, config):
        # setting configuration values 
        self.dimension_noise = 100
        self.beta1 = 0.5
        self.d_iterations = 5
        self.l1_val = 50
        self.l2_val = 100
        self.logger = Logger('gan')
        self.path_to_checkpoints = 'checkpoints'
        self.save_directory_path = config['save_directory_path']
        self.gan_type = config['gan_type']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.lr = config['lr']
        self.num_epochs = config['epochs']
        self.cls = config['cls']        
        self.generator = torch.nn.DataParallel(repository.generator_factory(self.gan_type).cuda())       
        self.discriminator = torch.nn.DataParallel(repository.discriminator_factory(self.gan_type).cuda())         

        if config['path_to_pre_trained_discriminator'] != '':
            self.discriminator.load_state_dict(torch.load(config['path_to_pre_trained_discriminator']))
        else:
            self.discriminator.apply(Tools.randomly_initialize_weights)

        if config['path_to_pre_trained_generator'] != '':
            self.generator.load_state_dict(torch.load(config['path_to_pre_trained_generator']))
        else:
            self.generator.apply(Tools.randomly_initialize_weights)
       
        self.optimG = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.optimD = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        
        self.dataset = CustomDataset(config['dataset_path'], split=config['split'])        
        
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_workers)

        