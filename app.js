/* ═══════════════════════════════════════════════════════════
   ASCII Studio — text → ASCII art, powered by FIGlet.js
   ═══════════════════════════════════════════════════════════ */

import figlet from "./vendor/figlet.mjs";

const $ = (id) => document.getElementById(id);

const els = {
  logo: $("logo"),
  fontCount: $("font-count"),
  text: $("text-input"),
  fontInput: $("font-input"),
  fontList: $("font-list"),
  fontPicker: $("font-picker"),
  prev: $("font-prev"),
  next: $("font-next"),
  random: $("font-random"),
  layout: $("h-layout"),
  width: $("width"),
  widthVal: $("width-val"),
  wrap: $("wrap"),
  size: $("preview-size"),
  sizeVal: $("size-val"),
  preview: $("preview"),
  status: $("preview-status"),
  theme: $("export-theme"),
  copy: $("copy"),
  txt: $("export-txt"),
  png: $("export-png"),
  svg: $("export-svg"),
  html: $("export-html"),
  toast: $("toast"),
};

const STORAGE_KEY = "ascii-studio-v1";

/** Curated styles pinned to the top of the picker. */
const POPULAR = [
  "Standard", "Big", "Slant", "Small", "Shadow", "ANSI Shadow",
  "Banner3-D", "Bloody", "Calvin S", "Colossal", "Doom", "Ghost",
  "Graffiti", "Isometric1", "Larry 3D", "Mini", "Script", "Small Script",
  "Speed", "Star Wars", "Sub-Zero", "Tinker-Toy",
];

const THEMES = {
  terminal: { bg: "#0d1117", fg: "#3fb950" },
  dark:     { bg: "#0d1117", fg: "#e6edf3" },
  light:    { bg: "#ffffff", fg: "#24292f" },
};

let FONTS = [];
let currentArt = "";
let renderToken = 0;
let renderTimer = null;
let listOpen = false;
let activeIndex = -1;
let visibleItems = [];

const state = {
  text: "Hello World",
  font: "Standard",
  layout: "default",
  width: 80,
  wrap: true,
  size: 12,
  theme: "terminal",
};

figlet.defaults({ fontPath: "fonts", fetchFontIfMissing: true });

/* ───────── state persistence ───────── */

function loadState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (saved && typeof saved === "object") Object.assign(state, saved);
  } catch { /* ignore */ }
}

function saveState() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch { /* ignore */ }
}

/* ───────── rendering ───────── */

function scheduleRender() {
  clearTimeout(renderTimer);
  renderTimer = setTimeout(render, 120);
}

async function render() {
  const token = ++renderToken;
  const text = els.text.value;

  if (!text.trim()) {
    currentArt = "";
    els.preview.classList.add("placeholder");
    els.preview.classList.remove("error");
    els.preview.textContent = "Your ASCII art will appear here — type something on the left!";
    els.status.textContent = "Waiting for input";
    return;
  }

  try {
    const art = await figlet.text(text, {
      font: state.font,
      horizontalLayout: state.layout,
      width: state.width,
      whitespaceBreak: state.wrap,
    });
    if (token !== renderToken) return; // stale result
    currentArt = art;
    els.preview.classList.remove("placeholder", "error");
    els.preview.textContent = art;
    const lines = art.replace(/\s+$/, "").split("\n");
    const cols = Math.max(...lines.map((l) => l.length));
    els.status.textContent = `${state.font} · ${cols} cols × ${lines.length} rows`;
  } catch (err) {
    if (token !== renderToken) return;
    els.status.textContent = `Couldn't render with "${state.font}"`;
    toast(`Failed to load font "${state.font}"`);
  }
}

/* ───────── font picker ───────── */

function openList() {
  if (!FONTS.length) return;
  listOpen = true;
  els.fontList.classList.add("open");
  els.fontInput.setAttribute("aria-expanded", "true");
  const q = els.fontInput.value === state.font ? "" : els.fontInput.value;
  renderList(q);
}

function closeList() {
  listOpen = false;
  els.fontList.classList.remove("open");
  els.fontInput.setAttribute("aria-expanded", "false");
  els.fontInput.value = state.font;
  activeIndex = -1;
}

function renderList(query) {
  els.fontList.textContent = "";
  visibleItems = [];
  const q = (query || "").trim().toLowerCase();
  const frag = document.createDocumentFragment();

  const addGroup = (label) => {
    const g = document.createElement("div");
    g.className = "group-label";
    g.textContent = label;
    frag.appendChild(g);
  };

  const addItem = (name) => {
    const d = document.createElement("div");
    d.className = "font-item";
    d.setAttribute("role", "option");
    if (name === state.font) d.classList.add("current");
    if (q) {
      const i = name.toLowerCase().indexOf(q);
      d.append(name.slice(0, i));
      const m = document.createElement("mark");
      m.textContent = name.slice(i, i + q.length);
      d.append(m, name.slice(i + q.length));
    } else {
      d.textContent = name;
    }
    d.addEventListener("mousedown", (e) => {
      e.preventDefault();
      selectFont(name);
    });
    frag.appendChild(d);
    visibleItems.push(name);
  };

  if (!q) {
    const pop = POPULAR.filter((f) => FONTS.includes(f));
    if (pop.length) {
      addGroup("Popular");
      pop.forEach(addItem);
    }
    addGroup(`All fonts (${FONTS.length})`);
    FONTS.forEach(addItem);
  } else {
    const matches = FONTS.filter((f) => f.toLowerCase().includes(q));
    if (!matches.length) {
      const n = document.createElement("div");
      n.className = "no-results";
      n.textContent = `No fonts match “${query.trim()}”`;
      frag.appendChild(n);
    } else {
      matches.forEach(addItem);
    }
  }

  els.fontList.appendChild(frag);
  activeIndex = visibleItems.indexOf(state.font);
  updateActive(false);
}

function moveActive(delta) {
  if (!visibleItems.length) return;
  activeIndex = (activeIndex + delta + visibleItems.length) % visibleItems.length;
  updateActive(true);
}

function updateActive(scroll) {
  const items = els.fontList.querySelectorAll(".font-item");
  items.forEach((el, i) => el.classList.toggle("active", i === activeIndex));
  if (scroll && items[activeIndex]) {
    items[activeIndex].scrollIntoView({ block: "nearest" });
  }
}

function selectFont(name) {
  state.font = name;
  els.fontInput.value = name;
  closeList();
  saveState();
  render();
}

function stepFont(delta) {
  if (!FONTS.length) return;
  const i = FONTS.indexOf(state.font);
  selectFont(FONTS[(i + delta + FONTS.length) % FONTS.length]);
}

function randomFont() {
  if (!FONTS.length) return;
  let name;
  do {
    name = FONTS[Math.floor(Math.random() * FONTS.length)];
  } while (name === state.font && FONTS.length > 1);
  selectFont(name);
}

/* ───────── copy & export helpers ───────── */

let toastTimer = null;
function toast(msg) {
  els.toast.textContent = msg;
  els.toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => els.toast.classList.remove("show"), 2000);
}

function slug() {
  const base = (els.text.value.trim() || "ascii-art")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 40);
  return base || "ascii-art";
}

function download(filename, blob) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}

function escXml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function exportGuard() {
  if (!currentArt) {
    toast("Nothing to export yet — type something first");
    return false;
  }
  return true;
}

async function copyArt() {
  if (!currentArt) return toast("Nothing to copy yet");
  try {
    await navigator.clipboard.writeText(currentArt);
    toast("Copied to clipboard ✓");
  } catch {
    const ta = document.createElement("textarea");
    ta.value = currentArt;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
    toast("Copied to clipboard ✓");
  }
}

function exportTxt() {
  if (!exportGuard()) return;
  download(`${slug()}.txt`, new Blob([currentArt], { type: "text/plain;charset=utf-8" }));
  toast("Downloaded .txt");
}

/** Build a self-contained SVG rendering of the ASCII art. */
function buildSvg(art, themeName, fontSize = 16) {
  const { bg, fg } = THEMES[themeName] || THEMES.terminal;
  const lines = art.replace(/\s+$/, "").split("\n");
  const lh = 1.25;
  const cw = fontSize * 0.601; // typical monospace advance width
  const maxCols = Math.max(1, ...lines.map((l) => l.length));
  const pad = Math.round(fontSize * 1.6);
  const w = Math.ceil(maxCols * cw + pad * 2);
  const h = Math.ceil(lines.length * fontSize * lh + pad * 2);
  const y0 = (pad + fontSize * 0.9).toFixed(1);
  const dy = (fontSize * lh).toFixed(2);
  const tspans = lines
    .map((l, i) => `<tspan x="${pad}"${i ? ` dy="${dy}"` : ""}>${escXml(l) || " "}</tspan>`)
    .join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><rect width="${w}" height="${h}" fill="${bg}"/><text x="${pad}" y="${y0}" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, 'Courier New', monospace" font-size="${fontSize}" fill="${fg}" xml:space="preserve">${tspans}</text></svg>`;
}

function exportSvg() {
  if (!exportGuard()) return;
  const svg = buildSvg(currentArt, els.theme.value);
  download(`${slug()}.svg`, new Blob([svg], { type: "image/svg+xml;charset=utf-8" }));
  toast("Downloaded .svg");
}

function exportPng() {
  if (!exportGuard()) return;
  const svg = buildSvg(currentArt, els.theme.value, 16);
  const url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml;charset=utf-8" }));
  const img = new Image();
  img.onload = () => {
    const scale = 2; // crisp 2× raster
    const canvas = document.createElement("canvas");
    canvas.width = img.width * scale;
    canvas.height = img.height * scale;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      URL.revokeObjectURL(url);
      if (blob) {
        download(`${slug()}.png`, blob);
        toast("Downloaded .png");
      } else {
        toast("PNG export failed");
      }
    }, "image/png");
  };
  img.onerror = () => {
    URL.revokeObjectURL(url);
    toast("PNG export failed");
  };
  img.src = url;
}

function exportHtml() {
  if (!exportGuard()) return;
  const { bg, fg } = THEMES[els.theme.value] || THEMES.terminal;
  const page = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escXml(slug())} — ASCII art</title>
<style>
  html, body { margin: 0; }
  body {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: ${bg};
  }
  pre {
    color: ${fg};
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Courier New", monospace;
    font-size: 14px;
    line-height: 1.25;
    padding: 32px;
    max-width: 100%;
    overflow: auto;
    box-sizing: border-box;
  }
</style>
</head>
<body>
<pre>${escXml(currentArt)}</pre>
</body>
</html>
`;
  download(`${slug()}.html`, new Blob([page], { type: "text/html;charset=utf-8" }));
  toast("Downloaded .html");
}

/* ───────── wiring ───────── */

function bindEvents() {
  els.text.addEventListener("input", () => {
    state.text = els.text.value;
    saveState();
    scheduleRender();
  });

  els.fontInput.addEventListener("focus", openList);
  els.fontInput.addEventListener("input", () => {
    if (!listOpen) openList();
    else renderList(els.fontInput.value);
  });
  els.fontInput.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") {
      if (!listOpen) openList();
      e.preventDefault();
      moveActive(1);
    } else if (e.key === "ArrowUp") {
      if (!listOpen) openList();
      e.preventDefault();
      moveActive(-1);
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (listOpen && activeIndex >= 0 && visibleItems[activeIndex]) {
        selectFont(visibleItems[activeIndex]);
      } else if (listOpen) {
        const q = els.fontInput.value.trim().toLowerCase();
        const match =
          FONTS.find((f) => f.toLowerCase() === q) ||
          FONTS.find((f) => f.toLowerCase().includes(q));
        if (match) selectFont(match);
        else closeList();
      }
    } else if (e.key === "Escape") {
      closeList();
    }
  });
  document.addEventListener("mousedown", (e) => {
    if (listOpen && !els.fontPicker.contains(e.target)) closeList();
  });

  els.prev.addEventListener("click", () => stepFont(-1));
  els.next.addEventListener("click", () => stepFont(1));
  els.random.addEventListener("click", randomFont);

  els.layout.addEventListener("change", () => {
    state.layout = els.layout.value;
    saveState();
    render();
  });

  els.width.addEventListener("input", () => {
    state.width = +els.width.value;
    els.widthVal.textContent = state.width;
    saveState();
    scheduleRender();
  });

  els.wrap.addEventListener("change", () => {
    state.wrap = els.wrap.checked;
    saveState();
    render();
  });

  els.size.addEventListener("input", () => {
    state.size = +els.size.value;
    els.sizeVal.textContent = `${state.size}px`;
    els.preview.style.fontSize = `${state.size}px`;
    saveState();
  });

  els.theme.addEventListener("change", () => {
    state.theme = els.theme.value;
    saveState();
  });

  els.copy.addEventListener("click", copyArt);
  els.txt.addEventListener("click", exportTxt);
  els.png.addEventListener("click", exportPng);
  els.svg.addEventListener("click", exportSvg);
  els.html.addEventListener("click", exportHtml);
}

/* ───────── init ───────── */

async function init() {
  loadState();

  try {
    const res = await fetch("fonts.json");
    if (!res.ok) throw new Error("font list unavailable");
    FONTS = await res.json();
  } catch {
    try { FONTS = figlet.fontsSync(); } catch { FONTS = ["Standard"]; }
  }

  els.fontCount.textContent = FONTS.length;
  if (!FONTS.includes(state.font)) {
    state.font = FONTS.includes("Standard") ? "Standard" : FONTS[0];
  }

  // Restore UI from state
  els.text.value = state.text;
  els.fontInput.value = state.font;
  els.layout.value = state.layout;
  els.width.value = state.width;
  els.widthVal.textContent = state.width;
  els.wrap.checked = state.wrap;
  els.size.value = state.size;
  els.sizeVal.textContent = `${state.size}px`;
  els.preview.style.fontSize = `${state.size}px`;
  els.theme.value = state.theme;

  bindEvents();
  render();

  // Render the ASCII banner in the header once the engine is warm.
  try {
    els.logo.textContent = await figlet.text("ASCII Studio", { font: "Slant" });
  } catch { /* keep the plain-text fallback */ }
}

init();
