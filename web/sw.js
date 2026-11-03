// Service worker for the installed app (Microsoft Store / "Install app").
// Network first, so the newest version always loads when online; the cached copy
// is used only when offline. Only this site's own files are handled: videos are
// read straight from disk, and the speech model's downloads (jsDelivr, Hugging
// Face) keep the browser's normal caching.
const CACHE = 'video-compressor-v1';
const SHELL = ['./', 'index.html', 'manifest.webmanifest', 'vendor/mp4box.all.min.js', 'vendor/mp4-muxer.js', 'icons/icon-192.png', 'icons/icon-512.png'];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET' || new URL(req.url).origin !== self.location.origin) return;
  event.respondWith(
    fetch(req)
      .then((res) => {
        if (res.ok && res.status === 200) {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(req, copy));
        }
        return res;
      })
      .catch(() => caches.match(req, { ignoreSearch: req.mode === 'navigate' }).then((hit) => hit || caches.match('index.html')))
  );
});
