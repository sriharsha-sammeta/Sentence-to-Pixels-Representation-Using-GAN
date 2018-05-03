# Sentence to Pixel Representations using GANs

Project by:
1. Venkata Sai Sriharsha Sammeta (vs2626)
2. Smitha Edakalavan (se2444)
3. Yuval Schaal (ys3055)

This code requires GPU and the following libraries needs to be installed. 

## Requirements
- pytorch 
- visdom
- numpy
- h5py
- PIL

## Individual Contributions

Venkata Sai Sriharsha Sammeta - vs2626
- CoreModels/infogan.py
- CoreModels/coremodel.py
- CoreModels/repository.py
- RNNModel/utility.py
- RNNModel/attentive_weights.py
- RNNModel/data.py
- RNNModel/model.py
- RNNModel/train.py
- CustomDatasetLoader.py
- helper.py
- main.py
- train_gan.py
- train_infogan.py

Smitha Edakalavan - se2444
- CoreModels/wgan.py
- train_wgan.py
- config.yaml
- logger.py

Yuval Schaal - ys3055
- CoreModels/dcgan.py
- train_dcgan.py
- scripts/script_hd5.py
- created figure 1 and figure 3 in reports shown in img/fig1 and img/fig3


## Code Organization
- main.py - It is the starting point of the code. It will read config.yaml and accordingly call the appropriate training/testing methods on corresponding gans
- {root directory} - Has Train_{gan_type} files like train_dcgan.py, train_wgan.py and train_infogan.py that are used to train / test the corresponding gan models
- CoreModels directory - Has all the models (DCGAN, WGAN and INFOGAN) in it
- RNNModel directory- Has the entire model and training process for the Attention Based RNN Embedding that we used 
- RNNModel/utility.py - has code for tools and helper methods used inside RNN model
- RNNModel/attentive_weights.py - code for evaluating the attention model
- RNNModel/data.py - has the code to load and modify corpus 
- RNNModel/model.py - Has the code which defines the architecture of the model
- RNNModel/train.py - Has the code required to train the RNN model
- scripts directory - Has the script required to convert the given data into hd5 format as required for pytorch
- CustomDataSetLoader.py - It is used for loading datasamples as required by the code via pytorch
- logger.py - has code for logging purpose
- helper.py - has code for some tools & visualization

## Quick Run Instructions
If you want to quickly test the implementation:
- Downaload this [file](https://drive.google.com/open?id=1EgnaTrlHGaqK5CCgHKLclZMT_AMSTyh8) which has flowers dataset in the format need for pytorch. 
- Put it in root directory
- run the command ``` python main.py```

## Detailed instructions to test various datasets and models
We are using 3 datasets to train our gan models. 

To download the dataset, please follow this for <a href='https://drive.google.com/file/d/1EgnaTrlHGaqK5CCgHKLclZMT_AMSTyh8/view'>Flowers</a>, <a href='https://drive.google.com/file/d/1mNhn6MYpBb-JwE86GC1kk0VJsYj-Pn5j/view'>Birds</a>, <a href='https://drive.google.com/file/d/0B0ywwgffWnLLLUc2WHYzM0Q2eWc/view?usp=sharing'>COCO</a> and put it in the root directory. 

Update `config.yaml` file with the above downloaded filename (already default is flowers.hd5):
``` shell
dataset: 'filename'
```

Likewise, we can switch between gan_type of dcgan, infogan and wgan by 
``` shell
gan_type: 'dcgan' or 'wgan' or 'infogan'
```

## Training 
To run the training process:
``` shell
python main.py
```

## Testing
To run the inference/testing:
In the config.yaml file, set inference: True.
Then run,
``` shell
python main.py
```
## Reference Links
- http://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/ 
- http://www.vision.caltech.edu/visipedia/CUB-200.html
- http://cocodataset.org/#home 
- https://github.com/paarthneekhara/text-to-image 
- https://github.com/zsdonghao/text-to-image <br>
- https://github.com/handspeaker/gan_practice/blob/master/wgan.py

