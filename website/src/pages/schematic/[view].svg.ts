import { schematic } from '../../scripts/schematic.mjs';

export function getStaticPaths() {
  return ['image', 'labels', 'ids'].map((view) => ({ params: { view } }));
}

export function GET({ params }: { params: { view: string } }) {
  return new Response(schematic(params.view), { headers: { 'Content-Type': 'image/svg+xml' } });
}
