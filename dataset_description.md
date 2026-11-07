# 2D Semantic Labeling Contest - Potsdam
The dataset contains 38 equal-sized tiles. Each pairs a true orthophoto (TOP)
from a larger mosaic with a digital surface model (DSM).

![potsdam_map](http://www2.isprs.org/tl_files/isprs/wg34/images/tile_overview.resized.png)
*Tile outlines on the orthophoto mosaic. The numbers match the filenames.*

Both products have 5 cm ground sampling distance. Trimble INPHO 5.6 generated
the DSM by dense image matching; OrthoVista generated the TOP mosaic. Tiles
come from the center of the mosaic. Small remaining gaps were interpolated.

The dataset lists three TOP TIFF variants with 8-bit channels:

- IRRG: infrared, red, green
- RGB: red, green, blue
- RGBIR: red, green, blue, infrared

DSM TIFFs store height as a single 32-bit float band. The dataset defines a
common WGS84/UTM grid for TOP and DSM. TIFF world files describe the affine
transform for each tile.

Normalized DSMs store height above the estimated terrain.

![image_dsm_gt](http://www2.isprs.org/tl_files/isprs/wg34/images/potsdam_top_dsm_label.png)

The contest provides labels for 24 tiles for training and internal evaluation.
The organizers use the remaining 14 tiles to score submissions.

![magevslabel](http://www2.isprs.org/tl_files/isprs/wg34/images/potsdam2D_table.png)
