/** Poster frame for a clip under /public/clips: same path with a .jpg extension. */
export function posterFor(src: string): string {
  return src.replace(/\.mp4$/, ".jpg");
}
