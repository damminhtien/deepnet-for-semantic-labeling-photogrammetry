export function composeHistoricalMappings(rgbPixels, infraredPixels) {
  if (rgbPixels.length === 0 || rgbPixels.length % 4 !== 0 || rgbPixels.length !== infraredPixels.length) {
    throw new RangeError('RGB and infrared samples must contain the same RGBA pixels.');
  }

  const observed = new Uint8ClampedArray(rgbPixels.length);
  const implied = new Uint8ClampedArray(rgbPixels.length);
  for (let offset = 0; offset < rgbPixels.length; offset += 4) {
    const nir = infraredPixels[offset];
    const red = rgbPixels[offset];
    const green = rgbPixels[offset + 1];
    const blue = rgbPixels[offset + 2];

    observed[offset] = nir;
    observed[offset + 1] = green;
    observed[offset + 2] = blue;
    observed[offset + 3] = 255;

    implied[offset] = nir;
    implied[offset + 1] = red;
    implied[offset + 2] = green;
    implied[offset + 3] = 255;
  }
  return { observed, implied };
}
