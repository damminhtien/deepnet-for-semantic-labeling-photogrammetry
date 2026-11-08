import original from '../../../artifacts/public/metrics/potsdam-benchmark-confusions.json?raw';

export function GET() {
  return new Response(original, { headers: { 'Content-Type': 'application/json' } });
}
