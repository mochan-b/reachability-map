import { useState } from 'react'
import { useHdf5 } from './hooks/useHdf5'
import Viewport from './components/Viewport'

export default function App() {
  const [source, setSource] = useState<File | string | null>(
    import.meta.env.DEV ? '/quick.h5' : null
  )
  const { data, loading, error } = useHdf5(source)

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
        <Viewport />
      </main>
    </div>
  )
}
