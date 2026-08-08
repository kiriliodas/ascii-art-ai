/* ═══════════════════════════════════════════════════════════
   Dot-matrix icons — Nothing-Phone style glyph grid.
   Each icon is an 11×11 bitmap ("#" = dot) rendered as SVG
   circles, so icons scale crisply and inherit `currentColor`.
   ═══════════════════════════════════════════════════════════ */

export const ICONS = {
  /* chevron-left */
  "chev-left": [
    "...........",
    "...........",
    "......##...",
    ".....##....",
    "....##.....",
    "...##......",
    "....##.....",
    ".....##....",
    "......##...",
    "...........",
    "...........",
  ],

  /* chevron-right */
  "chev-right": [
    "...........",
    "...........",
    "...##......",
    "....##.....",
    ".....##....",
    "......##...",
    ".....##....",
    "....##.....",
    "...##......",
    "...........",
    "...........",
  ],

  /* dice (random) */
  dice: [
    "###########",
    "#.........#",
    "#.........#",
    "#..#...#..#",
    "#.........#",
    "#....#....#",
    "#.........#",
    "#..#...#..#",
    "#.........#",
    "#.........#",
    "###########",
  ],

  /* copy — two overlapping pages */
  copy: [
    "#######....",
    "#.....#....",
    "#.....#....",
    "#.....#....",
    "#.....#####",
    "#.....#...#",
    "#######...#",
    "....#.....#",
    "....#.....#",
    "....#.....#",
    "....#######",
  ],

  /* download — arrow into tray */
  download: [
    ".....#.....",
    ".....#.....",
    ".....#.....",
    "#....#....#",
    ".#...#...#.",
    "..#..#..#..",
    "...#.#.#...",
    "....###....",
    ".....#.....",
    "#.........#",
    "###########",
  ],

  /* .txt — page with lines */
  "file-text": [
    ".#########.",
    ".#.......#.",
    ".#.#####.#.",
    ".#.......#.",
    ".#.#####.#.",
    ".#.......#.",
    ".#.#####.#.",
    ".#.......#.",
    ".#.###...#.",
    ".#.......#.",
    ".#########.",
  ],

  /* .png — framed picture */
  image: [
    "...........",
    "###########",
    "#.........#",
    "#..#......#",
    "#.........#",
    "#....#....#",
    "#...###...#",
    "#..#####..#",
    "#.#######.#",
    "###########",
    "...........",
  ],

  /* .svg — diamond with vector node */
  diamond: [
    ".....#.....",
    "....#.#....",
    "...#...#...",
    "..#.....#..",
    ".#.......#.",
    "#....#....#",
    ".#.......#.",
    "..#.....#..",
    "...#...#...",
    "....#.#....",
    ".....#.....",
  ],

  /* </> .html — code brackets */
  code: [
    "...........",
    "...........",
    "...........",
    "...##...##.",
    "..##....##.",
    ".##......##",
    "..##....##.",
    "...##...##.",
    "...........",
    "...........",
    "...........",
  ],

  /* check mark */
  check: [
    "...........",
    "...........",
    "...........",
    "........##.",
    ".......##..",
    "......##...",
    "#....##....",
    ".#..##.....",
    "..###......",
    "...#.......",
    "...........",
  ],
};

/**
 * Render a named icon as an inline SVG string of dots.
 * @param {string} name  key into ICONS
 * @param {number} size  rendered width/height in px
 */
export function iconSvg(name, size = 16) {
  const grid = ICONS[name];
  if (!grid) return "";
  const rows = grid.length;
  const cols = grid[0].length;
  const cell = 2; // viewBox units per grid cell
  const r = 0.72; // dot radius (< cell/2 keeps the gaps visible)
  let dots = "";
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      if (grid[y][x] === "#") {
        dots += `<circle cx="${x * cell + 1}" cy="${y * cell + 1}" r="${r}"/>`;
      }
    }
  }
  return (
    `<svg viewBox="0 0 ${cols * cell} ${rows * cell}" width="${size}" height="${size}" ` +
    `fill="currentColor" aria-hidden="true" focusable="false">${dots}</svg>`
  );
}

/** Prepend the matching icon to every element with a data-icon attribute. */
export function hydrateIcons(root = document) {
  root.querySelectorAll("[data-icon]").forEach((el) => {
    if (el.dataset.iconDone) return; // idempotent
    el.dataset.iconDone = "1";
    el.insertAdjacentHTML("afterbegin", iconSvg(el.dataset.icon));
  });
}
