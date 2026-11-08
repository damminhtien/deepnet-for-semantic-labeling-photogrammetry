# Benchmark metric formulas

Let $C_{ij}$ count scored pixels with reference class $i$ and predicted class $j$.
Let $N = \sum_{i,j} C_{ij}$. For each class $c$,
$\mathrm{TP}_c = C_{cc}$,
$\mathrm{FP}_c = \sum_{i \ne c} C_{ic}$, and
$\mathrm{FN}_c = \sum_{j \ne c} C_{cj}$.

Overall accuracy:

$$
\mathrm{OA} = \frac{\sum_c \mathrm{TP}_c}{N}
$$

Class intersection over union and its six-class mean:

$$
\mathrm{IoU}_c = \frac{\mathrm{TP}_c}{\mathrm{TP}_c + \mathrm{FP}_c + \mathrm{FN}_c}
$$

$$
\mathrm{mIoU} = \frac{1}{6}\sum_{c=0}^{5} \mathrm{IoU}_c
$$

Class F1:

$$
\mathrm{F1}_c = \frac{2\mathrm{TP}_c}{2\mathrm{TP}_c + \mathrm{FP}_c + \mathrm{FN}_c}
$$

The public Potsdam recomputation assigns zero to a class with a zero IoU or F1
denominator. Ignore ID 6 is excluded from scoring. For aggregate scores, sum the
scene matrices first and then evaluate the formulas. An average of per-scene
scores answers a different question.

See the [scoring implementation](../scripts/recompute_public_metrics.py)
and the [audited results](results.md).
