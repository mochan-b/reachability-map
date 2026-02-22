import { useState, useMemo } from 'react'
import * as THREE from 'three'
import { useControls, Leva } from 'leva'
import { useHdf5 } from './hooks/useHdf5'
import Viewport from './components/Viewport'
import ReachabilityVoxels from './components/ReachabilityVoxels'
import ClipPlaneWidget from './components/ClipPlaneWidget'
import { COLORMAP_NAMES, type ColormapName } from './lib/colormaps'
import RobotModel from './components/RobotModel'

// Initial clip plane: horizontal at origin, normal up (+Y in Three.js = +Z robot).
// Clips nothing by default since the robot workspace is above y = 0.
const INITIAL_PLANE = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)

export default function App() {
  const [source, setSource] = useState<File | string | null>(
    import.meta.env.DEV ? '/quick.h5' : null
  )
  const { data, loading, error } = useHdf5(source)

  const { threshold, colormap, opacity } = useControls('Display', {
    threshold: { value: 0.0, min: 0.0, max: 1.0, step: 0.01, label: 'Threshold' },
    colormap:  { options: COLORMAP_NAMES as unknown as string[], label: 'Colormap' },
    opacity:   { value: 0.8, min: 0.1, max: 1.0, step: 0.01, label: 'Opacity' },
  })

  const { showRobot, robotOpacity } = useControls('Robot', {
    showRobot:    { value: true,  label: 'Show robot' },
    robotOpacity: { value: 0.4, min: 0.05, max: 1.0, step: 0.01, label: 'Opacity' },
  })

  // Clip controls — axis sliders and free-plane flip are shown conditionally
  // via Leva's render option (hidden when the mode doesn't match).
  const { clipMode, clipX, clipY, clipZ, flipPlane } = useControls('Clip', {
    clipMode: {
      options: ['Off', 'Axis sliders', 'Free plane'] as unknown as string[],
      label: 'Mode',
    },
    clipX: {
      value: 3.0, min: -3.0, max: 3.0, step: 0.01, label: 'X cut',
      render: (get) => get('Clip.clipMode') === 'Axis sliders',
    },
    clipY: {
      value: 3.0, min: -3.0, max: 3.0, step: 0.01, label: 'Y cut',
      render: (get) => get('Clip.clipMode') === 'Axis sliders',
    },
    clipZ: {
      value: 3.0, min: -3.0, max: 3.0, step: 0.01, label: 'Z cut',
      render: (get) => get('Clip.clipMode') === 'Axis sliders',
    },
    flipPlane: {
      value: false, label: 'Flip',
      render: (get) => get('Clip.clipMode') === 'Free plane',
    },
  })

  // Plane state managed by the 3D widget (updated via drag)
  const [freePlane, setFreePlane] = useState<THREE.Plane>(() => INITIAL_PLANE.clone())

  // Coordinate mapping for axis mode (robot frame → Three.js world):
  //   three(x,y,z) = robot(x, z, y)
  //   clip robot.x ≤ V  →  plane normal(-1, 0, 0), const V
  //   clip robot.y ≤ V  →  three.z = robot.y  →  normal(0, 0, -1), const V
  //   clip robot.z ≤ V  →  three.y = robot.z  →  normal(0, -1, 0), const V
  const clipPlanes = useMemo<THREE.Plane[]>(() => {
    if (clipMode === 'Axis sliders') {
      return [
        new THREE.Plane(new THREE.Vector3(-1,  0,  0), clipX),
        new THREE.Plane(new THREE.Vector3( 0,  0, -1), clipY),
        new THREE.Plane(new THREE.Vector3( 0, -1,  0), clipZ),
      ]
    }
    if (clipMode === 'Free plane') {
      return [
        flipPlane
          ? new THREE.Plane(freePlane.normal.clone().negate(), -freePlane.constant)
          : freePlane,
      ]
    }
    return []
  }, [clipMode, clipX, clipY, clipZ, freePlane, flipPlane])

  const stats = data
    ? (() => {
        const r = data.reachabilityIndex
        let min = Infinity, max = -Infinity, reachable = 0
        for (let i = 0; i < r.length; i++) {
          if (r[i] < min) min = r[i]
          if (r[i] > max) max = r[i]
          if (r[i] > 0) reachable++
        }
        return { min, max, reachable, total: r.length }
      })()
    : null

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-gray-950 text-white">
      {/* Leva panel — floats in the top-right of the viewport */}
      <Leva />

      {/* Sidebar */}
      <aside className="w-64 flex-shrink-0 flex flex-col gap-4 p-4 bg-gray-950 border-r border-gray-800 overflow-y-auto">
        <h1 className="text-lg font-semibold tracking-tight">reachability-map</h1>

        <label className="cursor-pointer rounded border border-gray-600 px-3 py-1.5 text-sm text-center hover:border-gray-400 transition-colors">
          Load HDF5
          <input
            type="file"
            accept=".h5,.hdf5"
            className="hidden"
            onChange={(e) => setSource(e.target.files?.[0] ?? null)}
          />
        </label>

        {loading && <p className="text-gray-400 text-xs">Loading…</p>}
        {error   && <p className="text-red-400 text-xs">Error: {error}</p>}

        {data && stats && (
          <div className="font-mono text-xs text-gray-300 space-y-1">
            <p>Shape: [{data.gridShape.join(', ')}]</p>
            <p>Min:  {stats.min.toFixed(4)}</p>
            <p>Max:  {stats.max.toFixed(4)}</p>
            <p>Reachable: {stats.reachable} / {stats.total}</p>
            <p>Delta: {data.gridAttrs.delta} m</p>
            <p>Mode: {data.attrs['mode']}</p>
            <p>Robot: {data.attrs['robot_name']}</p>
            <p>IK: {data.attrs['ik_solver']}</p>
          </div>
        )}
      </aside>

      {/* 3D Viewport */}
      <main className="flex-1 relative">
        <Viewport>
          {data && (
            <ReachabilityVoxels
              data={data}
              threshold={threshold}
              colormap={colormap as ColormapName}
              opacity={opacity}
              clipPlanes={clipPlanes}
            />
          )}
          {clipMode === 'Free plane' && (
            <ClipPlaneWidget onPlaneChange={setFreePlane} />
          )}
          <RobotModel visible={showRobot} opacity={robotOpacity} />
        </Viewport>
      </main>
    </div>
  )
}
