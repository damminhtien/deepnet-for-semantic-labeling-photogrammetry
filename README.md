# DeepNet-for-Semantic-Labeling-Photogrammetry
This deep learning project is my graduated thesis which is scored 9.5/10. It is also the first time i do end-to-end an applied research project by myself.  

This project aim to automated extraction of urban objects from data acquired by airborne sensors which very high-resolution data (leads to high intra-class variance while the inter-class variance is low), focus is on detailed 2D semantic segmentation that assigns labels to multiple object categories. At the end, my accuraccy reachs 90% (the top world leader 92%)

# Dataset: 
* 2D Semantic Labeling Contest - [Potsdam](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html)

# Pipeline
1. Visualise and analysis data
2. Preprocessing data
3. Build model
4. Choose hyper-parameters and loss function
5. Experiment and validate
6. Testing phase
7. Report and submit

> ## 1. Visualise and analysis data

> ## 2. Preprocessing data
* Image size 6000x6000 5 channel (IR-R-G-DSM-nDSM), split to patchs 256x256
* Train/Valid/Test: 18/6/14

> ## 3. Build model
## Segnet
84%

## FCN-Resnet101
85%

## Unet101-Resnet101
89% (single model)

## PSPNet-Resnet101
90% (aulixiary loss) (single model)
> ## 4. Choose hyper-parameters and loss function
> ## 5. Experiment and validate
> ## 6. Testing phase
> ## 7. Report and submit

# Expriment


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
