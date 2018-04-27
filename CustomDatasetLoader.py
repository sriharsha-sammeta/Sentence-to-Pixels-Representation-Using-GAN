import os
import io
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import pdb
from PIL import Image
import torch
from torch.autograd import Variable
import pdb
import torch.nn.functional as F

#vs2626
class CustomDataset(Dataset):
    """A custom dataset loader to load data from the input into pytorch"""
    def __init__(self, pathToFile, split=0):
        """initialization"""
        self.pathToFile = pathToFile
        self.split = 'train' if split == 0 else 'valid' if split == 1 else 'test'
        self.h5py2int = lambda x: int(np.array(x))
        self.transform = None
        self.dataset = None
        self.dataset_keys = None

    def __getitem__(self, idx):
        """over loads getitem function of parent to customize loading of data by selecting specific values from input"""
        if self.dataset is None:
            self.dataset = h5py.File(self.pathToFile, mode='r')
            self.dataset_keys = [str(k) for k in self.dataset[self.split].keys()]

        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]

        # pdb.set_trace()

        CorrectImage = bytes(np.array(example['img']))
        CorrectEmbedding = np.array(example['embeddings'], dtype=float)
        IncorrectImage = bytes(np.array(self.find_IncorrectImage(example['class'])))
        IntermediateEmbedding = np.array(self.find_IntermediateEmbedding())

        CorrectImage = Image.open(io.BytesIO(CorrectImage)).resize((64, 64))
        IncorrectImage = Image.open(io.BytesIO(IncorrectImage)).resize((64, 64))

        CorrectImage = self.validate_image(CorrectImage)
        IncorrectImage = self.validate_image(IncorrectImage)

        txt = np.array(example['txt']).astype(str)

        sample = {
                'CorrectImages': torch.FloatTensor(CorrectImage),
                'CorrectEmbedding': torch.FloatTensor(CorrectEmbedding),
                'IncorrectImages': torch.FloatTensor(IncorrectImage),
                'IntermediateEmbedding': torch.FloatTensor(IntermediateEmbedding),
                'txt': str(txt)
                 }

        sample['CorrectImages'] = sample['CorrectImages'].sub_(127.5).div_(127.5)
        sample['IncorrectImages'] =sample['IncorrectImages'].sub_(127.5).div_(127.5)

        return sample

    def find_IncorrectImage(self, category):
        """find fake images"""
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        _category = example['class']

        if _category != category:
            return example['img']

        return self.find_IncorrectImage(category)

    def find_IntermediateEmbedding(self):
        """finds and capture intermediate embeddings"""
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        return example['embeddings']


    def validate_image(self, img):
        """validation of the image"""
        img = np.array(img, dtype=float)
        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb

        return img.transpose(2, 0, 1)


    def __len__(self):
        """overload len function to extract out keys"""
        myfile = h5py.File(self.pathToFile, 'r')
        self.dataset_keys = [str(k) for k in myfile[self.split].keys()]
        length = len(myfile[self.split])
        myfile.close()

        return length