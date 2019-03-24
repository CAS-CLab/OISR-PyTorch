# OISR+SENet : an example of OISR with attention
The PyTorch4.0.1 implementation of our NTIRE2019 model. Unfortunately, this model performs poorly on the real single-image super-resolution problem due to the gap between bicubic-downsampling and this new challenge. Hopefully, OISR modules can be used in the winner models of this competition to further improve the state-of-the-arts.

### Dependencies :
* Python 3.7
* PyTorch >= 0.4.0
* numpy
* skimage
* **imageio**
* matplotlib
* tqdm

### How to use ?
1. Download the pre-processed training and validation sets from [here](https://pan.baidu.com/s/1vG95V0g08lCnQ5K6EzhVUg) with code `eo20`.
2. Unzip images to the given folders:
```shell
unzip ./train_HR.zip -d /your/NTIRE2019/data/DIV2K/DIV2K_train_HR/
unzip ./train_LR.zip -d /your/NTIRE2019/data/DIV2K/DIV2K_train_LR_bicubic/
unzip ./val_LR.zip -d /your/NTIRE2019/data/DIV2K/DIV2K_test_LR_bicubic/
unzip ./val_LR.zip -d /your/NTIRE2019/data/DIV2K/DIV2K_train_LR_bicubic/
unzip ./val_HR.zip -d /your/NTIRE2019/data/DIV2K/DIV2K_train_HR/
```
3. Training from scratch:
```shell
cd /your/NTIRE2019/OISR/src
bash train.sh
```
4. Evaluation on validation set:
```shell
cd /your/NTIRE2019/OISR/src
cp ../experiment/OISR/model/model_best.pt ./
bash val.sh
```
5. Evaluation on test set:
```shell
# Download Test_LR.zip somewhere
zip ./Test_LR.zip -d /your/NTIRE2019/data/benckmark/B100/HR/
python test_images_rename.py
cd /your/NTIRE2019/OISR/src
cp ../experiment/OISR/model/model_best.pt ./ # or move the pre-trained model to ./
bash test.sh
python test_SRimages_rename.py # SR images can be found in ../experiment/test/results-B100
```

### Pre-trained models:
[Baidu Pan]()
[OneDrive]()
