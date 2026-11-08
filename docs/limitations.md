# Limitations and validity

## Internal validity

The DINOv3 RGB and PSPNet RGB runs share scene IDs, the recorded split hash, RGB cache,
eroded references, 512×512 windows, 256-pixel stride, batch size, and identity inference.
They still use different architectures, initialization, and training recipes. Their
score difference therefore belongs to the complete runs; it is not an architecture-only
effect. DINOv2 and SAM differ in additional input, decoder, training, and inference
choices. The exact DINOv2 reference raster identity was not re-established for the
current public bundle. The SAM export has no current split hash, although its listed
scene IDs and scored-pixel count match the cohort.

The A/B comparison groups infrared, a spectral index, DSM relief, and an availability
mask. It cannot identify the effect of any one feature. The C comparison changes several
augmentation probabilities and ranges together. Those results support statements about
the tested configurations and seeds only. The paired DINOv3/PSPNet inference was checked
against archived confusion matrices, but it does not add a new training or selection
experiment.

## External validity and benchmark reuse

The primary comparison uses 14 Potsdam benchmark scenes that have appeared in earlier
campaigns. They are a historical evaluation cohort, not an untouched or independent
holdout. Recomputing their scores once from a final checkpoint does not remove prior
exposure during experiment design. No independent city or acquisition is used to support
generalization claims.

The class mapping, ground sampling distance, sensor, and image conditions are specific
to the studied ISPRS data. Vaihingen appears in earlier probes and a separate joint
DINOv2 experiment; those runs use different protocols and are not a direct external
validation of the four-model Potsdam comparison.

## Statistical validity

The final benchmark contains one checkpoint per reported recipe. DINOv3 has a separate
seed-43 validation confirmation, but there are no repeated final benchmark seeds for
DINOv3 and PSPNet. Scene-level bootstrap intervals are descriptive summaries over 14
spatially related scenes; millions of pixels must not be treated as independent
observations. Per-scene and per-class tables expose heterogeneity but do not establish
population-level uncertainty.

## Label and metric scope

Training uses participant labels. Validation and benchmark scoring use eroded labels,
which exclude boundary pixels. Scores do not measure full-boundary accuracy. Clutter /
background is heterogeneous, and the available labels do not provide object-subtype
annotations. IoU, F1, and OA describe agreement with the supplied six-class reference;
they do not explain the cause of an error.

## Runtime and artifact scope

Scoring times are measurements from specific Kaggle profiles and hardware, not portable
latency guarantees. Total training GPU-hours depend on which configurations and failed
starts are included. The public repository includes per-scene confusion matrices and
selected metadata, but not checkpoints, prediction rasters, source imagery, or full
Kaggle output bundles.

## Licensing and use

The repository does not currently include a code license, so code reuse terms are
unspecified here. ISPRS imagery, labels, and pretrained DINOv3/SAM weights have their
own source terms. The public evidence bundle does not redistribute those assets and does
not change their terms. A citation file or code license should be added only after the
relevant authorship and distribution rights have been confirmed.
