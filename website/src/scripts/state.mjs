export const query = () => new URLSearchParams(location.search);

export function restore(root) {
  const params = query();
  for (const field of root.querySelectorAll('[data-share]')) {
    const value = params.get(field.name);
    if (value === null) continue;
    if (field instanceof HTMLSelectElement) {
      if (Array.from(field.options).some((option) => option.value === value)) field.value = value;
    } else if (field.type === 'range' && Number.isFinite(Number(value))) {
      field.value = value;
    }
  }
}

export function share(root, extra = {}) {
  const url = new URL(location.href);
  for (const field of root.querySelectorAll('[data-share]')) url.searchParams.set(field.name, field.value);
  for (const [key, value] of Object.entries(extra)) {
    if (value == null) url.searchParams.delete(key);
    else url.searchParams.set(key, value);
  }
  history.replaceState(null, '', url);
}
