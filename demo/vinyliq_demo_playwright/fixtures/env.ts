/** Shared Playwright env helpers for demo + pitch-assist specs. */

export function envOr(name: string, fallback: string): string {
  const v = process.env[name];
  return v && v.trim().length > 0 ? v : fallback;
}

export function envMs(name: string, fallback: number): number {
  const v = process.env[name];
  if (!v || v.trim().length === 0) {
    return fallback;
  }
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}
