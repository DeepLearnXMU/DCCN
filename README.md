# DCCN
The code for "Dynamic Context-guided Capsule Network for Multimodal Machine Translation" in Pytorch. 
This project is based on https://github.com/Waino/OpenNMT-py. 

## Installation
```python
pip install torch==0.3.1
pip install six
pip install tqdm
pip install torchtext==0.2.3
pip install future
```

## Quickstart
### Step 1: Preprocess the data
First, you can download Multi30k dataset [here](https://github.com/multi30k/dataset). The official dataset provides tokenized textual data, pre-extracted visual features (global visual features) and raw images, which are all required in the following steps. 

Next, run the following command to preprocess textual data:
```python
python preprocess.py -train_src src-train.txt -train_tgt tgt-train.txt -valid_src src-val.txt -valid_tgt tgt-val.txt -save_data demo
```

To extract regional visual features from raw images, you need to: 
1) Extract class annotation scores from raw image by using [tools/generate_tsv.py](https://github.com/peteanderson80/bottom-up-attention/blob/master/tools/generate_tsv.py). The cls_scores in func "get_detections_from_im" is the probability distribution of class annotations. Please set the MAX_BOXES=50 to exsure that no more than 50 regions are extracted in each image.
2) Since each image has different region numbers, you need to additionally save a mask file, of which the size is \[N_dataset, 50\]. Use 1 to indicate a masked position, otherwise 0. And then use ``np.save()``  to save them as **train/valid/test_obj_mask.npy**.
3) Download the annotation vocab [here](https://github.com/peteanderson80/bottom-up-attention/blob/master/data/genome/1600-400-20/objects_vocab.txt). 
4) Map the annotation vocab to MT vocab index (attribute id). If a word is splitted by BPE, it will be mapped to several subwords' index, and its prediction probability will also be divided equally. This will cause N_word bigger than original annotator vocab length (1600), so please change "1600" in onmt/modules/multimodal.py, L42 and L45 to the real N_word you obtain.  
5) Concatenate the probability vector and attribute id vector for each region of every image, and use ``np.save()`` to save them as **train/valid/test_obj.npy**.

![example](https://github.com/DeepLearnXMU/DCCN/blob/master/example.png)

For more instructions, please refer to [Installation](https://github.com/peteanderson80/bottom-up-attention#installation) and [Demo](https://github.com/peteanderson80/bottom-up-attention#demo) parts of [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention).


### Step 2: Train the model
```python
python train_mmod.py \
 -data demo \
 -save_model demo_modelname \
 -path_to_train_img_feats train-resnet50-res4frelu.npy \
 -path_to_valid_img_feats val-resnet50-res4frelu.npy \
 -path_to_train_attr train_obj.npy \
 -path_to_valid_attr val_obj.npy \
 -path_to_train_img_mask train_obj_mask.npy \
 -path_to_valid_img_mask val_obj_mask.npy \
 -encoder_type transformer -decoder_type transformer --multimodal_model_type dcap
```
Here, ``-path_to_train/valid_img_feats`` refers to global visual features, ``-path_to_train/valid_attr`` refers to regional visual feretures.


### Step 3: Translate sentences
```python
python translate_mmod.py \
 -model demo_modelname.pt \
 -src test_src.txt -output text_tgt.txt \
 -path_to_test_img_feats test-resnet50-res4frelu.npy \
 -path_to_test_attr test_obj.npy \
 -path_to_test_img_mask test_obj_mask.npy \
 -replace_unk -verbose --multimodal_model_type dcap
```
Here, ``-path_to_test_img_feats`` refers to global visual features, ``-path_to_test_attr`` refers to regional visual feretures.
