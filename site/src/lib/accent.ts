import type { Accent } from "@/content/types";

/**
 * Accent colours as raw values so they can be used in inline styles.
 * Tailwind cannot generate classes from runtime strings, so per-entry accents
 * flow through CSS custom properties instead of utility classes.
 */
export const ACCENT: Record<Accent, { hex: string; rgb: string }> = {
  orange: { hex: "#f97316", rgb: "249,115,22" },
  blue: { hex: "#3b82f6", rgb: "59,130,246" },
  red: { hex: "#e11d48", rgb: "225,29,72" },
  green: { hex: "#22c55e", rgb: "34,197,94" },
  violet: { hex: "#8b5cf6", rgb: "139,92,246" },
  sky: { hex: "#38bdf8", rgb: "56,189,248" },
  neutral: { hex: "#c9c9d1", rgb: "201,201,209" },
};

/** Inline style object that sets --entry-accent and --entry-accent-rgb. */
export function accentVars(accent: Accent): React.CSSProperties {
  const a = ACCENT[accent];
  return {
    "--entry-accent": a.hex,
    "--entry-accent-rgb": a.rgb,
  } as React.CSSProperties;
}
