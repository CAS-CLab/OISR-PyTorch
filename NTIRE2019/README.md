# OISR+CA : an example of OISR with attention
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
1. Download the pre-processed training and validation sets from [Baidu Cloud](https://pan.baidu.com/s/1wU6EWMUbAgasJVBGjxTa0w) with code `ij6f` or from [OneDrive](https://1drv.ms/u/s!Av1MQK8mV3J8gmz3FKBlKa1SZ3BY).
2. Unzip images to the given folders:
```shell
unzip /your/download/data.zip -d /your/NTIRE2019/
```
3. Training from scratch:
```shell
cd /your/NTIRE2019/OISR/src
bash train.sh
```
4. Fine-tuning on patches, download the pre-processed training patches and validation sets from [Baidu Cloud](https://pan.baidu.com/s/1y5FQMYe96hqiuv3a0KTvZQ) with code `mu0a` or from [OneDrive](https://1drv.ms/u/s!Av1MQK8mV3J8gmtownoDhLJBwHy5):
```shell
unzip /your/download/data2.zip -d /your/NTIRE2019/
cd /your/NTIRE2019/OISR/src/
cp ../experiment/OISR/model/model_best.pt ./
bash train2.sh
```
5. Evaluation on test set, download `Test_LR.zip` from [Baidu Cloud](https://pan.baidu.com/s/1-eQFiO-nj5btDI8ym7yWbA) with code `fwnh` or from [OneDrive](https://1drv.ms/u/s!Av1MQK8mV3J8gm2Xko-mFzwgRVIo):
```shell
zip /your/download/Test_LR.zip -d /your/NTIRE2019/data/benckmark/B100/
cd /your/NTIRE2019/OISR/src
cp ../experiment/OISR/model/model_best.pt ./ # or move the pre-trained model to ./
bash test.sh
python test_SRimages_rename.py # SR images can be found in ../experiment/test/results-B100
```

### Model Structure:
![](./OISR_AC.jpg)
In this case, we apply the channel attention module (similar to SE/CBAM) to the RK-3 block.

### Pre-trained models:
[Baidu Pan]()
[OneDrive]()
