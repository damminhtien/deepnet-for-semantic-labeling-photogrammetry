# DeepNet-for-Semantic-Labeling-Photogrammetry
Automated extraction of urban objects from data acquired by airborne sensors which very high-resolution data,  leads to high intra-class variance while the inter-class variance is low, focus is on detailed 2D semantic segmentation that assigns labels to multiple object categories.
# Dataset: 
* 2D Semantic Labeling Contest - [Potsdam](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html)
* Image size 6000x6000 5 channel (IR-R-G-DSM-nDSM), split to patchs 256x256
* Train/Valid/Test: 18/6/14
# Expriment
## Segnet

## FCN-Resnet101

## Unet101-Resnet101

## PSPNet-Resnet101

### `More`
* Use cyclical learning rate
* Use Resnet101 pretrain
* Split to maxable size patch
* Choose maxable batch size (6->32)
* Adam optimizer
* Small weight decay lr
* Use less pooling
* Use more skip connection
* Normalize image before training

### Guide to run
Install package and run notebook on your computer (with anaconda, or cloud such as kaggle, colab)
