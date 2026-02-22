/** [t, r, g, b] — all values normalised 0–1 */
type Stop = [number, number, number, number]

function buildLut(stops: Stop[]): Float32Array {
  const lut = new Float32Array(256 * 3)
  for (let i = 0; i < 256; i++) {
    const t = i / 255
    // Find the lower bracketing stop
    let lo = 0
    for (let j = 0; j < stops.length - 1; j++) {
      if (stops[j][0] <= t) lo = j
    }
    const hi = Math.min(lo + 1, stops.length - 1)
    const [t0, r0, g0, b0] = stops[lo]
    const [t1, r1, g1, b1] = stops[hi]
    const a = t1 > t0 ? (t - t0) / (t1 - t0) : 0
    lut[i * 3 + 0] = r0 + a * (r1 - r0)
    lut[i * 3 + 1] = g0 + a * (g1 - g0)
    lut[i * 3 + 2] = b0 + a * (b1 - b0)
  }
  return lut
}

// Key stops sourced from matplotlib's canonical colormap definitions
const VIRIDIS_STOPS: Stop[] = [
  [0.000, 0.267, 0.005, 0.329],
  [0.125, 0.282, 0.140, 0.457],
  [0.250, 0.230, 0.321, 0.545],
  [0.375, 0.172, 0.475, 0.558],
  [0.500, 0.128, 0.567, 0.551],
  [0.625, 0.157, 0.661, 0.493],
  [0.750, 0.369, 0.789, 0.383],
  [0.875, 0.678, 0.864, 0.189],
  [1.000, 0.993, 0.906, 0.144],
]

const PLASMA_STOPS: Stop[] = [
  [0.000, 0.050, 0.030, 0.528],
  [0.125, 0.258, 0.008, 0.615],
  [0.250, 0.448, 0.003, 0.657],
  [0.375, 0.614, 0.091, 0.620],
  [0.500, 0.741, 0.218, 0.525],
  [0.625, 0.847, 0.351, 0.408],
  [0.750, 0.938, 0.498, 0.258],
  [0.875, 0.985, 0.663, 0.092],
  [1.000, 0.940, 0.976, 0.131],
]

const JET_STOPS: Stop[] = [
  [0.000, 0.000, 0.000, 0.500],
  [0.100, 0.000, 0.000, 1.000],
  [0.350, 0.000, 1.000, 1.000],
  [0.500, 0.500, 1.000, 0.500],
  [0.650, 1.000, 1.000, 0.000],
  [0.900, 1.000, 0.000, 0.000],
  [1.000, 0.500, 0.000, 0.000],
]

const GRAYSCALE_STOPS: Stop[] = [
  [0.0, 0.0, 0.0, 0.0],
  [1.0, 1.0, 1.0, 1.0],
]

const LUTS: Record<string, Float32Array> = {
  viridis:   buildLut(VIRIDIS_STOPS),
  plasma:    buildLut(PLASMA_STOPS),
  jet:       buildLut(JET_STOPS),
  grayscale: buildLut(GRAYSCALE_STOPS),
}

export const COLORMAP_NAMES = ['viridis', 'plasma', 'jet', 'grayscale'] as const
export type ColormapName = (typeof COLORMAP_NAMES)[number]

/** Sample a colormap at t ∈ [0, 1]. Returns [r, g, b] each in [0, 1]. */
export function sampleColormap(name: ColormapName, t: number): [number, number, number] {
  const lut = LUTS[name] ?? LUTS['viridis']
  const i = Math.max(0, Math.min(255, Math.round(t * 255)))
  return [lut[i * 3], lut[i * 3 + 1], lut[i * 3 + 2]]
}

/** Return the precomputed 256×3 RGB Float32Array for a colormap. */
export function getColormapLut(name: ColormapName): Float32Array {
  return LUTS[name] ?? LUTS['viridis']
}
