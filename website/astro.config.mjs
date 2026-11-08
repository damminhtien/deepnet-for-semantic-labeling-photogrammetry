import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://damminhtien.github.io',
  base: '/deepnet-for-semantic-labeling-photogrammetry',
  trailingSlash: 'always',
  output: 'static',
  devToolbar: { enabled: false },
});
