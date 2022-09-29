# DeepNet-for-Semantic-Labeling-Photogrammetry

This project aim to automated extraction of urban objects from data acquired by airborne sensors which very high-resolution data (leads to high intra-class variance while the inter-class variance is low), focus is on detailed 2D semantic segmentation that assigns labels to multiple object categories. At the end, my accuraccy reachs 90% (the top world leader at  92%)

# Dataset: 
* 2D Semantic Labeling Contest - [Potsdam](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html)
* Short description at [here](https://github.com/damminhtien/deepnet-for-semantic-labeling-photogrammetry/blob/master/dataset_description.md)

# Pipeline
1. Visualise and analyse data
2. Preprocessing data
3. Build model
4. Choose hyper-parameters and loss function
5. Experiment and validate
6. Testing phase
7. Report and submit

> ## 1. Visualise and analyse data
See the [detail](https://github.com/damminhtien/deepnet-for-semantic-labeling-photogrammetry/blob/master/Insight-data-potsdam.ipynb).
Conclusion:
* Unbalance dataset: building, road and low vegetable class have highest acreage, while car class is negligible. 
* Large file size, high resolution.
* The tree class looks like low veg. class on the RGB images. So we use additional DSM channel to distinguish.
* The clutter class 's (disregard class, not the main class) percentage is quite big.

> ## 2. Preprocessing data
* Image size 6000x6000 5 channel (IR-R-G-DSM-nDSM), split to patchs 256x256
* Train/Valid/Test: 18/6/14

> ## 3 & 4. Build model and Experiment
I tried lots of models, but there are 3 main:

| My Models                                                                | Origin Paper     | My customize code                                                                                                                                   | Acc  |
|--------------------------------------------------------------------------|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|------|
| Fully Convolutional Networks with Resnet 101 encoder, 3 skip connections | [FCN paper]()    | [My customize implement](https://github.com/damminhtien/deepnet-for-semantic-labeling-photogrammetry/blob/master/model_script/fcn-resnet101-3sc.py) | 0.85 |
| Unet with Resnet 101 encoder                                             | [Unet paper]()   | [My customize implement]()                                                                                                                          | 0.89 |
| Pyramid Spatial Pooling Network with Resnet 101 encoder, aulixiary loss  | [PSPNet paper]() | [My customize implement]()                                                                                                                          | 0.90 |                                                                                                                      |
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
