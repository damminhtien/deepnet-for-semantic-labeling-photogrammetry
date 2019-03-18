# DeepNet-for-Semantic-Labeling-Photogrammetry
Automated extraction of urban objects from data acquired by airborne sensors which very high-resolution data,  leads to high intra-class variance while the inter-class variance is low, focus is on detailed 2D semantic segmentation that assigns labels to multiple object categories.
# Dataset: 
2D Semantic Labeling Contest - [Potsdam](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html)
# Expriment
## Segnet Vgg16
* 58% (after 25 epochs)
## FCN Resnet101
* Lr 0.01 
* Adam Optimizer
* Crit CategoricalCross Entropy
* Batchsize 10
-> ~80%
* After 90 epochs
* Total accuracy : 87.04862724541366%
* Kappa: 0.7952076079553785 (Deeper net get higher acc at building but low acc at tree, car)
## FCN8s VGG16_bn
* After 40 epochs
* Total accuracy : 84.97403337091006%
* Kappa: 0.799597872523977
## FCN8s VGG19_bn
* After 40 epochs
* Total accuracy : 82.30135182217282%
* Kappa: 0.7630174827777142
