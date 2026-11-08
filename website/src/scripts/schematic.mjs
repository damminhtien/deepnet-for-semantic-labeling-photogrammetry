import { colors } from './metrics.mjs';

export const SIZE = 512;
const fill = ['#676f69', '#b8886c', '#8c9a69', '#354f43', '#edc390', '#b0a797'];
const roof = ['#b68e73', '#978775', '#c8ae93', '#a17d68', '#a29a86'];
const svg = (tag, attrs) => `<${tag} ${attrs}/>`;
const rect = (x, y, w, h, color, attrs = '') => svg('rect', `x="${x}" y="${y}" width="${w}" height="${h}" fill="${color}" ${attrs}`);
const circle = (x, y, r, color, attrs = '') => svg('circle', `cx="${x}" cy="${y}" r="${r}" fill="${color}" ${attrs}`);

function city(view) {
  let seed = 318;
  const random = () => ((seed = (Math.imul(1664525, seed) + 1013904223) >>> 0) / 2 ** 32);
  const palette = view === 'image' ? fill : view === 'ids'
    ? colors.map((_, i) => `rgb(${30 + i * 40},0,0)`) : colors;
  const roads = [0, 112, 228, 365, 475, 592];
  let ground = rect(-190, -140, 900, 820, palette[0]);
  let details = '';
  for (let row = 0; row < 5; row++) {
    for (let col = 0; col < 6; col++) {
      const x = -112 + col * 112 + (row % 2) * 11;
      const y = roads[row] - 30;
      const w = 94, h = roads[row + 1] - roads[row] - 20;
      const park = (row * 5 + col * 11) % 7 < 2;
      ground += rect(x, y, w, h, palette[park ? 2 : 5], 'rx="5"');
      if (!park) {
        ground += rect(x + 44, y + 6, 44, h - 12, palette[2]);
        ground += rect(x + 13, y + h - 22, 47, 12, palette[0]);
      }
      for (let i = 0; i < (park ? 34 : 14); i++) {
        const cx = x + 9 + random() * 77, cy = y + 9 + random() * (h - 18);
        const r = 2 + random() * 6;
        if (view === 'image') ground += circle(cx + 3, cy + 3, r + 1, '#222f2d', 'opacity=".42"');
        ground += circle(cx, cy, r, view === 'image' && i % 2 ? '#56674e' : palette[3]);
      }
      for (let i = 0; i < (park ? 1 : 4); i++) {
        const bx = x + 8 + i % 2 * 40 + random() * 5;
        const by = y + 10 + Math.floor(i / 2) * (h / 2) + random() * 4;
        const bw = 19 + random() * 13, bh = 17 + random() * (h / 4);
        const tone = roof[Math.floor(random() * roof.length)];
        if (view === 'image') ground += rect(bx + 4, by + 5, bw + 2, bh + 2, '#242f2f', 'opacity=".62"');
        ground += rect(bx, by, bw, bh, view === 'image' ? tone : palette[1]);
        if (view === 'image') {
          ground += rect(bx + bw * .52, by + 1, bw * .46, bh - 2, '#362d2a', 'opacity=".21"');
          ground += rect(bx + 2, by + 2, bw - 4, 1, '#fbdbab', 'opacity=".49"');
          ground += rect(bx + bw * .5, by + 4, 1, bh - 8, '#e5c0a3', 'opacity=".58"');
        }
      }
      for (let i = 0; i < 3; i++) {
        const cx = x + 12 + random() * 64, cy = y + h + 3 + i % 2 * 7;
        ground += rect(cx, cy, 6, 3, palette[4], 'rx="1"');
      }
      if (view === 'image') details += svg('path', `d="M${x-4} ${y+h+13}h${w+12}" stroke="#e6d9b8" opacity=".45" stroke-dasharray="5 7" fill="none"`);
    }
  }
  if (view === 'image') {
    details += '<path d="M-30 80Q100 245 562 399" fill="none" stroke="#203b3e" stroke-width="25" opacity=".74"/><path d="M-30 80Q100 245 562 399" fill="none" stroke="#687f72" stroke-width="17"/>';
  } else {
    details += `<path d="M-30 80Q100 245 562 399" fill="none" stroke="${palette[5]}" stroke-width="25"/>`;
  }
  return `<g transform="rotate(-17 256 256)">${ground}${details}</g>`;
}

export function schematic(view) {
  const texture = view === 'image' ? '<defs><pattern id="texture" width="7" height="9" patternUnits="userSpaceOnUse"><path d="M1 2h1m3 3h1m-4 3h1" stroke="#f4e6cf" opacity=".22" stroke-width=".6"/></pattern></defs><rect width="512" height="512" fill="url(#texture)"/>' : '';
  return `<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">${city(view)}${texture}</svg>`;
}
