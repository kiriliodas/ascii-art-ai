# ASCII Studio

Turn any text into ASCII art, right in the browser. Type your text, pick a
style from **328 FIGlet fonts**, tweak spacing and width, then copy the art or
export it.

## Features

- ⌨️ **Live preview** — art re-renders as you type (multi-line supported)
- 🔤 **328 font styles** — searchable picker with a "Popular" shortlist,
  ◀ ▶ browsing and a 🎲 random button
- 🛠 **Style controls** — letter spacing (full / fitted / smushed), max line
  width, word-boundary wrapping
- 📋 **One-click copy** to the clipboard
- ⬇️ **Export** to `.txt`, `.png` (2× resolution), `.svg` (crisp, scalable)
  and `.html` (standalone page)
- 🎨 **Export themes** — Terminal green, Dark, Light
- 💾 Settings (text, font, layout…) persist in `localStorage`
- 🔌 **100% client-side** — no backend, no build step, no tracking

## Run it locally

Any static file server works:

```bash
python3 server.py            # http://localhost:8000
# or
npx serve .
```

> Note: opening `index.html` directly via `file://` will not work — fonts are
> loaded with `fetch()`, which needs http(s).

## How it works

The site is a small ES-module app ([`app.js`](app.js)) on top of
[FIGlet.js](https://github.com/patorjk/figlet.js) (vendored in
[`vendor/`](vendor)), which implements the full FIGfont spec. All 328 `.flf`
fonts from the FIGlet community collection are bundled in [`fonts/`](fonts)
and lazy-loaded on first use. PNG/SVG exports are generated client-side by
rendering the art into an SVG `<text>` block (and rasterising it to a canvas
for PNG).

## License

The app code is MIT. FIGlet.js and the bundled fonts carry their own licenses —
see [`vendor/FIGLET-LICENSE.txt`](vendor/FIGLET-LICENSE.txt); most FIGlet fonts
are freely redistributable (see each font's comment header).
