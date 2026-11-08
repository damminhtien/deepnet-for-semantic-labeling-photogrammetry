export const colors = ['#9facad', '#4e8ab9', '#8cb587', '#316953', '#e6af66', '#ba726e'];
export const shortClasses = ['Impervious', 'Building', 'Low veg.', 'Tree', 'Car', 'Clutter'];
export const modelKeys = ['dinov3', 'pspnet', 'dinov2', 'sam'];
export const percent = (value, digits = 2) => `${(value * 100).toFixed(digits)}%`;
export const count = (value) => value.toLocaleString('en-US');

export function metrics(matrix) {
  const rows = matrix.map((row) => row.reduce((sum, value) => sum + value, 0));
  const total = rows.reduce((sum, value) => sum + value, 0);
  const correct = matrix.reduce((sum, row, i) => sum + row[i], 0);
  const iou = matrix.map((row, i) => {
    const predicted = matrix.reduce((sum, other) => sum + other[i], 0);
    const union = rows[i] + predicted - row[i];
    return union ? row[i] / union : 0;
  });
  return { rows, total, errors: total - correct, oa: correct / total,
    miou: iou.reduce((a, b) => a + b) / matrix.length, iou };
}

export function cropPixels(caseStudy, model, truth, predicted) {
  const column = model === 'dinov3' ? 1 : 2;
  return caseStudy.transitions.reduce((sum, transition) =>
    sum + (transition[0] === truth && transition[column] === predicted ? transition[3] : 0), 0);
}
