import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { deflateSync } from 'node:zlib';
import { metrics, cropPixels } from '../src/scripts/metrics.mjs';
import { readCrop } from '../src/scripts/crop-package.mjs';

const evidence = JSON.parse(readFileSync(new URL('../src/generated/evidence.json', import.meta.url)));
const { benchmark, cases } = evidence;
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} differs from ${expected}`);

test('browser recomputes archived metrics and vegetation error share', () => {
  for (const model of Object.values(benchmark.models)) {
    const summary = metrics(model.aggregate_confusion_matrix);
    assert.equal(summary.total, benchmark.scored_pixels_expected);
    close(summary.oa, model.overall_accuracy_source);
    close(summary.miou, model.mean_iou_source);
  }
  const dino = benchmark.models.dinov3.aggregate_confusion_matrix;
  const share = (dino[2][3] + dino[3][2]) / metrics(dino).errors * 100;
  assert.equal(share.toFixed(2), '30.37');
});

test('directed crop counts read the selected model, not the other prediction plane', () => {
  const crop = cases.find((item) => item.scene === '2_13' && item.row === 2048 && item.col === 512);
  assert.ok(crop);
  assert.notEqual(cropPixels(crop, 'dinov3', 2, 3), cropPixels(crop, 'pspnet', 2, 3));
  for (const model of ['dinov3', 'pspnet']) {
    let pixels = 0;
    for (let t = 0; t < 6; t++) for (let p = 0; p < 6; p++) pixels += cropPixels(crop, model, t, p);
    assert.equal(pixels, crop.valid_pixels);
  }
});

test('local loader accepts aligned class planes and rejects changes to evidence binding', async () => {
  const crop = { scene: 'test', row: 0, col: 0, height: 1, width: 2, transitions: [[2, 3, 2, 1]] };
  const { transitions, ...geometry } = crop;
  const packageFile = (overrides = {}) => new Blob([JSON.stringify({
    ...geometry, schema: 1, split_sha256: benchmark.split_sha256,
    label_protocol: benchmark.label_protocol, classes: benchmark.class_names,
    checkpoints: Object.fromEntries(['dinov3', 'pspnet'].map((key) => [key, benchmark.models[key].checkpoint_sha256])),
    planes: deflateSync(Buffer.from([2, 6, 3, 6, 2, 6])).toString('base64'),
    ...overrides,
  })]);
  const result = await readCrop(packageFile(), [crop], benchmark);
  assert.deepEqual(Array.from(result.planes), [2, 6, 3, 6, 2, 6]);
  await assert.rejects(readCrop(packageFile({ split_sha256: 'different' }), [crop], benchmark), /provenance/);
  await assert.rejects(readCrop(packageFile({ planes: deflateSync(Buffer.from([2, 6, 2, 6, 3, 6])).toString('base64') }), [crop], benchmark), /joint counts/);
});
