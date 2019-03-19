# DeepNet-for-Semantic-Labeling-Photogrammetry
Automated extraction of urban objects from data acquired by airborne sensors which very high-resolution data,  leads to high intra-class variance while the inter-class variance is low, focus is on detailed 2D semantic segmentation that assigns labels to multiple object categories.
# Dataset: 
2D Semantic Labeling Contest - [Potsdam](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html)
# Expriment
* Train image:  ['2_10', '2_11', '3_10', '4_12', '5_10', '6_7', '6_10', '7_9', '7_10']
* Test image: ['2_12', '3_11', '3_12', '5_12']
## Segnet
* Iter/epoch: 12000
* Window size: 224x224
* Cross Entropy Loss
* Adam Optimizer (weight_decay=0.005) 

| Epochs     | BatchSize | BaseLR (x0.1 / n epochs) | TestTime/epoch | Acc  | Kappa |
|------------|-----------|--------------------------|----------------|------|-------|
| 0 -> 100   | 10        | 0.01 [10, 20, 30, 40]    |                | 53   | 42    |
| **100 ->150**  | **8**         | **0.0005 [10, 20, 30, 40]**  |                | **54.5** | **44**   |
| 150 -> 200 | 5         | 0.0005 [10, 20, 30, 40]  |                | 54.3 | 43.7  |
| 200 -> 250 | 5         | 0.0001 [10, 20, 30, 40]  |                | 53.5 | 42.9  |
### F1 score :
* roads: 0.6558699847022574
* buildings: 0.6513313152517916
* low veg.: 0.6049431502694531
* trees: 0.518076807121743
* cars: 0.1654795648502439
* clutter: 0.08714544541088251
### Confusion matrix
|           | roads    | buildings | low veg. | trees    | cars    | clutter |
|-----------|----------|-----------|----------|----------|---------|---------|
| roads     | 19299357 | 1055747   | 2636265  | 1308289  | 5060542 | 5974458 |
| buildings | 1142158  | 20301265  | 1835422  | 3898711  | 5305991 | 2705301 |
| low veg.  | 1914579  | 4782290   | 19854174 | 5384168  | 2620030 | 3501172 |
| trees     | 909186   | 759929    | 2831696  | 11156876 | 2427509 | 2889157 |
| cars      | 41212    | 57141     | 38419    | 30309    | 1628656 | 78783   |
| clutter   | 210017   | 192532    | 387412   | 317649   | 766826  | 812563  |
## FCN Resnet101
* Lr 0.01 
* Adam Optimizer
* Crit CategoricalCross Entropy
* Batchsize 10
-> ~80%
* After 90 epochs
* Total accuracy : 82.21362315195232%
* Kappa: 0.7600396832725298 (Deeper net get higher acc at building but low acc at tree, car)
## FCN8s VGG16_bn
* After 40 epochs
* Total accuracy : 84.97403337091006%
* Kappa: 0.799597872523977
## FCN8s VGG19_bn
* After 40 epochs
* Total accuracy : 82.30135182217282%
* Kappa: 0.7630174827777142
## FCN8s VGG19
* After 40 epochs
### F1 score
* roads: 0.8661849732399588
* buildings: 0.9341539292198715
* low veg.: 0.6371709864269042
* trees: 0.30768265263939926
* cars: 0.7025409129162437
* clutter: 0.030470541857798166
* Total accuracy : 82.56446028790151%
* Kappa: 0.7661919835031442
