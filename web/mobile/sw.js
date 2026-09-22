/* Offline support: app shell, model and road databases are cached on first load;
 * map tiles are cached as they are viewed (so a planned route works in a tunnel). */
const SHELL = 'drishtinav-shell-v2';
const TILES = 'drishtinav-tiles-v1';
const ASSETS = ['/app/', '/app/app.js', '/app/app.css', '/app/engine.js', '/app/manifest.json', '/app/icon.svg', '/config.js',
  '/vendor/leaflet.js', '/vendor/leaflet.css', '/models/speednet.json', '/data/osm/index.json', '/data/osm/delhi_central.json',
  '/data/scenarios.json'];

self.addEventListener('install', e => e.waitUntil(caches.open(SHELL).then(c => c.addAll(ASSETS)).then(() => self.skipWaiting())));
self.addEventListener('activate', e => e.waitUntil(
  caches.keys().then(keys => Promise.all(keys.filter(k => k !== SHELL && k !== TILES).map(k => caches.delete(k))))
    .then(() => self.clients.claim())));

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (url.hostname.endsWith('tile.openstreetmap.org')) {
    e.respondWith(caches.open(TILES).then(async c => {
      const hit = await c.match(e.request);
      if (hit) return hit;
      const res = await fetch(e.request);
      if (res.ok) c.put(e.request, res.clone());
      return res;
    }));
    return;
  }
  if (url.origin === location.origin && !url.pathname.startsWith('/api/')) {
    e.respondWith(fetch(e.request).then(res => {
      if (res.ok) caches.open(SHELL).then(c => c.put(e.request, res.clone()));
      return res;
    }).catch(() => caches.match(e.request)));
  }
});
