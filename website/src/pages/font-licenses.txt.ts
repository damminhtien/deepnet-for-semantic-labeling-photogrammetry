import plexMono from '../../node_modules/@fontsource/ibm-plex-mono/LICENSE?raw';
import plexSans from '../../node_modules/@fontsource/ibm-plex-sans/LICENSE?raw';
import sourceSerif from '../../node_modules/@fontsource/source-serif-4/LICENSE?raw';

export function GET() {
  const notices = [['IBM Plex Mono', plexMono], ['IBM Plex Sans', plexSans], ['Source Serif 4', sourceSerif]];
  return new Response(notices.map(([name, text]) => `${name}\n\n${text}`).join('\n\n'), {
    headers: { 'Content-Type': 'text/plain; charset=utf-8' },
  });
}
