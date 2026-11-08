# Controlled pipeline study design

**Status: CONFIGURED_NOT_RUN** — this is a research design, not a completed experiment.

## Question

Which data preparation, modality, and augmentation choices change semantic segmentation
quality when model architecture, initialization, data split, sampling, training budget,
and inference are held fixed?

## Recommended sequence

1. **Optical band identity.** Compare only source-verified channel mappings using the
   same RGB-capable model and exact scene inputs. Record a mapping audit before training.
2. **Auxiliary features.** Compare RGB, RGB+IR, RGB+DSM, and RGB+IR+DSM with one fixed
   fusion interface. Lock the DSM registration and missing-data mask before the run.
3. **Augmentation.** Change one augmentation family at a time while preserving seed,
   samples, optimizer updates, and all other recipe settings.
4. **Seed replication.** Repeat the selected controlled contrasts on multiple seeds and
   report paired per-scene differences. Scenes, not pixels, are the spatial sampling
   units for descriptive uncertainty.
5. **Geographic evaluation.** Select a city or acquisition not used in data preparation
   decisions, model selection, or threshold tuning. Freeze ontology mapping and metrics
   before obtaining its labels.

## Reporting rules

For each contrast, publish the protocol, configuration hashes, seed, confusion matrices,
per-scene metrics, valid-pixel support, compute profile, and excluded-data rules. If
several factors change together, describe the result as a package comparison. Do not
interpret a metric difference as a causal effect unless the paired control isolates the
factor in question.

No additional training is scheduled by this design. The completed evidence and its
limits are summarized in [`research.md`](../research.md) and
[`limitations.md`](../limitations.md).
