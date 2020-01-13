# 2D Semantic Labeling Contest - Potsdam
The data set contains 38 patches (of the same size), each consisting of a true orthophoto (TOP) extracted from a larger TOP mosaic, see Figure below and a DSM. 

An orthophoto (orthophotograph or orthoimage) is an aerial photograph or satellite imagery geometrically corrected ("orthorectified") such that the scale is uniform: the photo or image has follows a given map projection. Unlike an uncorrected aerial photograph, an orthophoto can be used to measure true distances, because it is an accurate representation of the Earth's surface, having been adjusted for topographic relief,lens distortion, and camera tilt.

![potsdam_map](http://www2.isprs.org/tl_files/isprs/wg34/images/tile_overview.resized.png)
*Outlines of all patches given in Tab.1 overlaid with the true orthophoto mosaic. Numbers refer to the individual patch numbers, encoded in the filenames as well*

The ground sampling distance of both, the TOP and the DSM, is 5 cm. The DSM was generated via dense image matching with Trimble INPHO 5.6 software and Trimble INPHO OrthoVista was used to generate the TOP mosaic. In order to avoid areas without data (“holes”) in the TOP and the DSM, the patches were selected from the central part of the TOP mosaic and none at the boundaries. Remaining (very small) holes in the TOP and the DSM were interpolated.

The TOP come as TIFF files in different channel composistions, where each channel has a spectral resolution of 8bit:
* IRRG: 3 channels (IR-R-G)
* RGB: 3 channels (R-G-B)
* RGBIR: 4 channels (R-G-B-IR)

In this way participants can pick the data needed conveniently.

The DSM are TIFF files with one band; the grey levels (corresponding to the DSM heights) are encoded as 32 bit float values. The TOP and the DSM are defined on the same grid (UTM WGS84). Each tile comes with an affine transformation file (tiff world file) in order to enable a re-composition of images to larger mosaics if desired.

Normalised DSMs, that is, after ground filtering the ground height is removed for each pixel, leading to an representation of heights above the terrain. 

![image_dsm_gt](http://www2.isprs.org/tl_files/isprs/wg34/images/potsdam_top_dsm_label.png)

Labelled ground truth is provided for only one part of the data. The ground truth of the remaining scenes will remain unreleased and stays with the benchmark test organizers to be used for evaluation of submitted results. Participants shall use all data with ground truth for training (see Table) or internal evaluation of their method.

![magevslabel](http://www2.isprs.org/tl_files/isprs/wg34/images/potsdam2D_table.png)
