/** Read a local crop without transferring its bytes, and replay the published joint counts. */
export async function readCrop(file, cases, benchmark) {
  if (file.size > 2 * 1024 * 1024) throw new Error('Crop package exceeds 2 MB.');
  const pack = JSON.parse(await file.text());
  const crop = cases.find((candidate) => ['scene', 'row', 'col', 'height', 'width']
    .every((key) => candidate[key] === pack[key]));
  if (!crop || pack.schema !== 1 || pack.split_sha256 !== benchmark.split_sha256 ||
      pack.label_protocol !== benchmark.label_protocol ||
      JSON.stringify(pack.classes) !== JSON.stringify(benchmark.class_names) ||
      !['dinov3', 'pspnet'].every((key) => pack.checkpoints?.[key] === benchmark.models[key].checkpoint_sha256)) {
    throw new Error('The crop geometry or benchmark provenance does not match the public record.');
  }
  const area = crop.height * crop.width;
  const bytes = Uint8Array.from(atob(pack.planes), (character) => character.charCodeAt(0));
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate')).getReader();
  const planes = new Uint8Array(area * 3);
  let offset = 0;
  try {
    while (true) {
      const { done, value } = await stream.read();
      if (done) break;
      if (offset + value.length > planes.length) throw new Error('Crop contains too many pixels.');
      planes.set(value, offset);
      offset += value.length;
    }
  } finally {
    await stream.cancel();
  }
  if (offset !== planes.length) throw new Error('Crop is truncated.');

  const joint = new Uint32Array(216);
  for (let i = 0; i < area; i++) {
    const t = planes[i], d = planes[i + area], p = planes[i + 2 * area];
    if (t === 6 && d === 6 && p === 6) continue;
    if (t >= 6 || d >= 6 || p >= 6) throw new Error('Crop contains invalid class IDs.');
    joint[(t * 6 + d) * 6 + p]++;
  }
  for (const [t, d, p, n] of crop.transitions) {
    const index = (t * 6 + d) * 6 + p;
    if (joint[index] !== n) throw new Error('Crop does not reproduce published joint counts.');
    joint[index] = 0;
  }
  if (joint.some((n) => n !== 0)) throw new Error('Crop has unreported class transitions.');
  return { crop, planes };
}
