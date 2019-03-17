# DeepNet-for-Semantic-Labeling-Photogrammetry
Automated extraction of urban objects from data acquired by airborne sensors which very high-resolution data,  leads to high intra-class variance while the inter-class variance is low, focus is on detailed 2D semantic segmentation that assigns labels to multiple object categories.
# Dataset: 
2D Semantic Labeling Contest - [Potsdam](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html)
# Expriment
## Segnet Vgg16
1. Fine tune 1
* Lr 0.01
* SGD Optimizer
* Crit NNLoss (pytorch)
* Batchsize 10

-> All predict into only 2 class (Unbalance data?) 

-> fixed by Mixed loss (dice loss + focal loss)

2. Fine tune 2
* Lr 0.01 
* Adam Optimizer
* Crit MixedLoss
* Batchsize 15
* Train with random crop of first half dataset

-> 40% (after 25 epochs)
* Lr 0.001
* Train with random crop of second half dataset

-> 58% (after 25 epochs)
## FCN Resnet101
* Lr 0.01 
* Adam Optimizer
* Crit CategoricalCross Entropy
* Batchsize 10
-> ~80%
