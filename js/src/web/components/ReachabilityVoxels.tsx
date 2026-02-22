import { useRef, useMemo, useEffect } from 'react'
import { useThree } from '@react-three/fiber'
import * as THREE from 'three'
import type { ReachabilityData } from '../lib/hdf5Reader'
import { sampleColormap, type ColormapName } from '../lib/colormaps'

interface Props {
  data: ReachabilityData
  threshold: number
  colormap: ColormapName
  opacity: number
  clipPlanes: THREE.Plane[]
}

// Scratch objects — reused across renders to avoid GC pressure
const _mat = new THREE.Matrix4()
const _col = new THREE.Color()

export default function ReachabilityVoxels({ data, threshold, colormap, opacity, clipPlanes }: Props) {
  const meshRef = useRef<THREE.InstancedMesh>(null!)
  const { reachabilityIndex, voxelCenters, gridAttrs } = data
  const { gl } = useThree()

  // Three.js requires this flag on the renderer for per-material clipping planes
  useEffect(() => { gl.localClippingEnabled = true }, [gl])

  // ── Build sorted index arrays once per dataset ───────────────────────────
  // Indices of all non-zero voxels, sorted by reachability descending.
  // This lets us control the visible set with a single mesh.count update:
  //   mesh.count = K  →  shows the K voxels with highest reachability (≥ threshold)
  const { sortedIndices, sortedRi } = useMemo(() => {
    const indices: number[] = []
    for (let i = 0; i < reachabilityIndex.length; i++) {
      if (reachabilityIndex[i] > 0) indices.push(i)
    }
    indices.sort((a, b) => reachabilityIndex[b] - reachabilityIndex[a])
    return {
      sortedIndices: indices,
      sortedRi: Float32Array.from(indices, (i) => reachabilityIndex[i]),
    }
  }, [reachabilityIndex])

  const maxCount = sortedIndices.length

  // ── Visible count for current threshold (binary search, O(log N)) ────────
  const visibleCount = useMemo(() => {
    // sortedRi is descending; find the first position where value drops below threshold
    let lo = 0, hi = sortedRi.length
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (sortedRi[mid] >= threshold) lo = mid + 1
      else hi = mid
    }
    return lo
  }, [sortedRi, threshold])

  // ── Set instance matrices once per dataset ───────────────────────────────
  // Coordinate mapping: robot frame is Z-up; Three.js is Y-up.
  //   three(x, y, z) = robot(x, z, y)  [−90° rotation about X]
  useEffect(() => {
    const mesh = meshRef.current
    if (!mesh) return
    for (let j = 0; j < sortedIndices.length; j++) {
      const i = sortedIndices[j]
      _mat.setPosition(
        voxelCenters[i * 3 + 0],   // robot x → three x
        voxelCenters[i * 3 + 2],   // robot z → three y (up)
        voxelCenters[i * 3 + 1],   // robot y → three z
      )
      mesh.setMatrixAt(j, _mat)
    }
    mesh.instanceMatrix.needsUpdate = true
  }, [sortedIndices, voxelCenters])

  // ── Update instance colours when dataset or colormap changes ─────────────
  useEffect(() => {
    const mesh = meshRef.current
    if (!mesh) return
    for (let j = 0; j < sortedIndices.length; j++) {
      const [r, g, b] = sampleColormap(colormap, sortedRi[j])
      _col.setRGB(r, g, b)
      mesh.setColorAt(j, _col)
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  }, [sortedIndices, sortedRi, colormap])

  // ── Apply threshold by slicing the visible count ─────────────────────────
  useEffect(() => {
    const mesh = meshRef.current
    if (!mesh) return
    mesh.count = visibleCount
    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  }, [visibleCount])

  if (maxCount === 0) return null

  return (
    // key=maxCount forces a remount (and buffer reallocation) when the dataset changes
    <instancedMesh key={maxCount} ref={meshRef} args={[undefined, undefined, maxCount]}>
      <boxGeometry args={[gridAttrs.delta, gridAttrs.delta, gridAttrs.delta]} />
      <meshStandardMaterial
        clippingPlanes={clipPlanes}
        transparent={opacity < 0.99}
        opacity={opacity}
        depthWrite={opacity >= 0.99}
      />
    </instancedMesh>
  )
}
