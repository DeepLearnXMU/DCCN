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
To preprocess text data, run:
```python
python preprocess.py -train_src src-train.txt -train_tgt tgt-train.txt -valid_src src-val.txt -valid_tgt tgt-val.txt -save_data demo
```

To preorocess image data, please refer to https://github.com/peteanderson80/bottom-up-attention.

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
