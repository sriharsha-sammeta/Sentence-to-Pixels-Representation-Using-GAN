# Sentence to Pixels Representations using GANs

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


## Quick Run Instructions
If you want to quickly test the implementation:
- Downaload the href='https://drive.google.com/file/d/1EgnaTrlHGaqK5CCgHKLclZMT_AMSTyh8/view'>Flowers</a>.hd5 file
- Put it in root directory
- run the command ``` python main.py```

## Detailed intructions to test various datasets and models
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

