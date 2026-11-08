const base = import.meta.env.BASE_URL.replace(/\/$/, '');
export const route = (path = '') => `${base}/${path ? `${path}/` : ''}`;
export const repo = 'https://github.com/damminhtien/deepnet-for-semantic-labeling-photogrammetry';
export const source = (path, revision = 'master') => `${repo}/blob/${revision}/${path}`;
