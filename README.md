# DeepNet-for-Semantic-Labeling-Photogrammetry

This project explores 2D semantic segmentation of high-resolution airborne imagery. The original project notes report 90% accuracy; this checkout does not include the run metadata or checkpoint needed to reproduce that result.

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
* Unbalanced dataset: impervious surfaces (including roads), buildings, and low vegetation cover the largest areas, while cars are relatively rare.
* Large file size, high resolution.
* The tree class looks like low veg. class on the RGB images. So we use additional DSM channel to distinguish.
* The clutter class 's (disregard class, not the main class) percentage is quite big.

> ## 2. Preprocessing data
* Input tiles are 6000x6000 with five channels (IR-R-G-DSM-nDSM). Most models use 256x256 patches; the SegNet notebook uses 224x224.
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
Run a notebook with this repository as the working directory so the `semantic_labeling` helpers are importable. The raster paths still point to external ISPRS/Kaggle data directories; change the notebook path templates when your data is stored elsewhere. A Kaggle or Colab runtime must also have this repository's Python files available, not only the notebook.

The repository does not pin Python or ML dependency versions. The notebooks require a compatible environment with NumPy, scikit-image, scikit-learn, Matplotlib, tqdm, PyTorch, and torchvision. Training and inference have not been run as part of the code consistency fixes.
