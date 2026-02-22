import { useState } from 'react'
import { useHdf5 } from './hooks/useHdf5'

export default function App() {
  const [source, setSource] = useState<File | string | null>(
    import.meta.env.DEV ? '/quick.h5' : null
  )
  const { data, loading, error } = useHdf5(source)

  // Compute stats from flat Float32Array
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
    <div className="flex flex-col h-screen w-screen items-center justify-center gap-6 bg-gray-900 text-white">
      <h1 className="text-2xl font-semibold tracking-tight">reachability-map</h1>

      <label className="cursor-pointer rounded border border-gray-600 px-4 py-2 text-sm hover:border-gray-400 transition-colors">
        Load HDF5 file
        <input
          type="file"
          accept=".h5,.hdf5"
          className="hidden"
          onChange={(e) => setSource(e.target.files?.[0] ?? null)}
        />
      </label>

      {loading && <p className="text-gray-400 text-sm">Loading…</p>}
      {error && <p className="text-red-400 text-sm">Error: {error}</p>}

      {data && stats && (
        <div className="font-mono text-sm text-gray-300 space-y-1 text-left">
          <p>Shape: [{data.gridShape.join(', ')}]</p>
          <p>Min: {stats.min.toFixed(4)}  Max: {stats.max.toFixed(4)}</p>
          <p>Reachable: {stats.reachable} / {stats.total}</p>
          <p>Delta: {data.gridAttrs.delta}</p>
          <p>Mode: {data.attrs['mode']}</p>
          <p>Robot: {data.attrs['robot_name']}</p>
          <p>Orientations: {data.attrs['n_orientations']}</p>
        </div>
      )}
    </div>
  )
}
