import { ready, File as H5File, Dataset, Group, type OutputData } from 'h5wasm'

export interface ReachabilityData {
  reachabilityIndex: Float32Array   // flat [nx*ny*nz]
  voxelCenters: Float32Array        // flat [nx*ny*nz*3]
  orientations: Float32Array        // flat [N*4] qxyzw
  gridShape: [number, number, number]
  gridAttrs: { origin: number[]; delta: number }
  attrs: Record<string, string | number>  // version, urdf_path, mode, etc.
  poses?: Float32Array              // flat [M*8] full6d only
}

// Initialise WASM once for the lifetime of the module
let _modulePromise: Promise<Awaited<typeof ready>> | null = null
function getModule() {
  if (!_modulePromise) _modulePromise = ready
  return _modulePromise
}

/** Coerce an HDF5 attribute value to a plain JS string or number. */
function scalarAttr(v: OutputData | null): string | number {
  if (v === null) return ''
  if (typeof v === 'string' || typeof v === 'number') return v
  if (typeof v === 'bigint') return Number(v)
  if (v instanceof BigInt64Array || v instanceof BigUint64Array) return Number(v[0])
  if (ArrayBuffer.isView(v)) {
    const first = (v as unknown as ArrayLike<number | bigint>)[0]
    return typeof first === 'bigint' ? Number(first) : (first as number)
  }
  return String(v)
}

/** Return a detached Float32Array copy safe to keep after file.close(). */
function requireFloat32(data: OutputData | null, name: string): Float32Array {
  if (data instanceof Float32Array) return data.slice()
  if (data instanceof Float64Array) return new Float32Array(data)
  throw new Error(`Dataset "${name}": expected Float32/64, got ${(data as object)?.constructor?.name ?? typeof data}`)
}

/**
 * Read a reachability HDF5 file and return a typed ReachabilityData object.
 *
 * @param source  Browser File object (drag-drop / file input) or URL string to fetch.
 */
export async function readHdf5File(source: File | string): Promise<ReachabilityData> {
  const mod = await getModule()
  const FS = mod.FS

  let buffer: ArrayBuffer
  if (source instanceof File) {
    buffer = await source.arrayBuffer()
  } else {
    const res = await fetch(source)
    if (!res.ok) throw new Error(`Failed to fetch "${source}": HTTP ${res.status}`)
    buffer = await res.arrayBuffer()
  }

  // Write into the WASM virtual filesystem, open, read, clean up
  const tmpPath = '/reach_tmp.h5'
  FS.writeFile(tmpPath, new Uint8Array(buffer))

  const f = new H5File(tmpPath, 'r')
  try {
    // Root attrs
    const attrs: Record<string, string | number> = {}
    for (const [k, attr] of Object.entries(f.attrs)) {
      attrs[k] = scalarAttr(attr.value)
    }

    // grid/ group attrs
    const grid = f.get('grid') as Group
    const ga = grid.attrs
    const origin = Array.from(ga['origin'].value as Float32Array) as number[]
    const delta = scalarAttr(ga['delta'].value) as number
    const shapeRaw = ga['shape'].value as BigInt64Array
    const gridShape: [number, number, number] = [
      Number(shapeRaw[0]),
      Number(shapeRaw[1]),
      Number(shapeRaw[2]),
    ]

    // Datasets
    const reachabilityIndex = requireFloat32(
      (f.get('grid/reachability_index') as Dataset).value,
      'grid/reachability_index'
    )
    const voxelCenters = requireFloat32(
      (f.get('grid/voxel_centers') as Dataset).value,
      'grid/voxel_centers'
    )
    const orientations = requireFloat32(
      (f.get('orientations/samples') as Dataset).value,
      'orientations/samples'
    )

    // poses group is only present in full6d mode
    const posesEntity = f.get('poses/reachability_stats')
    const poses = posesEntity
      ? requireFloat32((posesEntity as Dataset).value, 'poses/reachability_stats')
      : undefined

    return { reachabilityIndex, voxelCenters, orientations, gridShape, gridAttrs: { origin, delta }, attrs, poses }
  } finally {
    f.close()
    FS.unlink(tmpPath)
  }
}
