import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { deflateSync } from 'node:zlib';
import { metrics, cropPixels } from '../src/scripts/metrics.mjs';
import { readCrop } from '../src/scripts/crop-package.mjs';
import { createHash } from 'node:crypto';

const evidence = JSON.parse(readFileSync(new URL('../src/generated/evidence.json', import.meta.url)));
const { benchmark, cases } = evidence;
const visualEvidence = JSON.parse(readFileSync(new URL('../src/generated/visual-evidence.json', import.meta.url)));
const pipelineTrace = JSON.parse(readFileSync(new URL('../src/generated/pipeline-trace.json', import.meta.url)));
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

test('published figure panels keep source checksums, crop identity, and checkpoint binding', () => {
  assert.equal(visualEvidence.assets.length, 5);
  assert.equal(visualEvidence.case_studies.length, 20);
  for (const asset of visualEvidence.assets) {
    const published = readFileSync(new URL('../public/' + asset.published_asset_path, import.meta.url));
    const source = readFileSync(new URL('../../docs/research/figures/' + asset.file, import.meta.url));
    const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
    assert.equal(digest(source), asset.source_artifact_sha256);
    assert.equal(digest(published), asset.published_asset_sha256);
    assert.equal(digest(source), digest(published));
    assert.equal(asset.distribution_rights_status, 'project_owner_authorized_curated_derivative');
    assert.match(asset.rights_review_url, /^https:\/\/www\.isprs\.org\//);
  }

  const grouped = new Map();
  const panelSizes = new Set();
  for (const crop of visualEvidence.case_studies) {
    const box = crop.pixel_window;
    const cropId = `${crop.scene_id}-${box.row}-${box.col}`;
    assert.equal(crop.case_id, cropId);
    assert.equal(crop.models.dinov3.checkpoint_sha256, benchmark.models.dinov3.checkpoint_sha256);
    assert.equal(crop.models.pspnet.checkpoint_sha256, benchmark.models.pspnet.checkpoint_sha256);
    assert.equal(crop.label_protocol, benchmark.label_protocol);
    assert.equal(crop.channel_mapping.join('/'), 'red/green/blue');
    assert.equal(crop.transitions.reduce((sum, row) => sum + row[3], 0), box.valid_pixels);
    for (const row of crop.transitions) {
      assert.ok(row.slice(0, 3).every((classId) => Number.isInteger(classId) && classId >= 0 && classId < 6));
      assert.ok(Number.isInteger(row[3]) && row[3] > 0);
    }
    const panels = Object.values(crop.panels);
    assert.equal(panels.length, 5);
    assert.ok(panels.every(({ width, height }) => [411, 412].includes(width) && height === 411));
    for (const { width, height } of panels) panelSizes.add(`${width}x${height}`);
    assert.ok(panels.every((panel) => panel.x >= 0 && panel.y >= 0 && panel.width > 0 && panel.height > 0));
    assert.ok(panels.every((panel) => panel.x + panel.width <= visualEvidence.assets.find((asset) => asset.asset_id === crop.figure_asset_id).width_px));
    grouped.set(crop.figure_asset_id, (grouped.get(crop.figure_asset_id) ?? 0) + 1);
  }
  assert.deepEqual([...grouped.values()].sort(), [4, 4, 4, 4, 4]);
  assert.deepEqual([...panelSizes].sort(), ['411x411', '412x411']);
});

test('pipeline trace binds real assets to one explicitly in-sample training draw', () => {
  assert.equal(pipelineTrace.schema_version, 1);
  assert.equal(pipelineTrace.sample.split_role, 'final_refit_training');
  assert.equal(pipelineTrace.sample.used_for_model_training, true);
  assert.equal(pipelineTrace.sample_confusion.scope, 'single_refit_training_draw_only');
  assert.deepEqual(pipelineTrace.stages.map((stage) => stage.id), [
    'source', 'preprocessing', 'sampling', 'augmentation', 'model-input',
    'activations', 'prediction', 'evaluation',
  ]);
  assert.deepEqual(pipelineTrace.model.feature_blocks, [3, 6, 9, 12]);

  for (const asset of Object.values(pipelineTrace.assets)) {
    const bytes = readFileSync(new URL(`../public/evidence/pipeline-trace/${asset.file}`, import.meta.url));
    assert.equal(createHash('sha256').update(bytes).digest('hex'), asset.sha256);
  }

  const matrix = pipelineTrace.sample_confusion.confusion_matrix_reference_rows_prediction_columns;
  const total = matrix.flat().reduce((sum, value) => sum + value, 0);
  const errors = matrix.reduce((sum, row, truth) => sum + row.reduce((inner, value, predicted) => inner + (truth === predicted ? 0 : value), 0), 0);
  assert.equal(total, pipelineTrace.sample_confusion.scored_pixels);
  assert.equal(errors, pipelineTrace.sample_confusion.error_pixels);
});
