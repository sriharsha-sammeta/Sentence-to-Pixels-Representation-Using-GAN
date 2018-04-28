import numpy as np
import h5py
import cv2

with open('config.yaml', 'r') as f:
	config = yaml.load(f)


images_path = config['images_path']
embedding_path = config['embedding_path']
text_path = config['text_path']
datasetDir = config['dataset_path']

f = h5py.File(datasetDir, 'w')
train = f.create_group('train')
valid = f.create_group('valid')
test = f.create_group('test')

for _class in sorted(hd5_path):
    if _class in open(config['train_split_path']).read().splitlines():
		split = train
	elif _class in open(config['val_split_path']).read().splitlines():
    		split = valid
	elif _class in open(config['test_split_path']).read().splitlines():
		split = test
    
   
# open a hdf5 file and create earrays
hdf5_file = h5py.File(hdf5_path, mode='w')
for example in (data_path + "/*.t7")):
	img_path = example['img']
	embeddings = example['embeddings'].numpy()
	example_name = img_path.split('/')[-1][:-4]

	img = cv2.imread(img_path)
	embeddings = embeddings[txt_choice]
	txt = np.array(example['txt'])
	txt = txt[txt_choice]
	

	hdf5_file.create_group(split)
        hdf5_file.create_dataset("embeddings", embeddings)
        hdf5_file.create_dataset("txt", txt)
	hdf5_file.create_dataset("txt", txt)
      
hdf5_file.close()