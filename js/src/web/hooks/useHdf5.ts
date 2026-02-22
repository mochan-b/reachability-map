import { useState, useEffect } from 'react'
import { readHdf5File, type ReachabilityData } from '../lib/hdf5Reader'

export interface UseHdf5Result {
  data: ReachabilityData | null
  loading: boolean
  error: string | null
}

/**
 * Load and parse a reachability HDF5 file.
 *
 * @param source  Browser File object, URL string, or null to clear.
 */
export function useHdf5(source: File | string | null): UseHdf5Result {
  const [data, setData] = useState<ReachabilityData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!source) {
      setData(null)
      setLoading(false)
      setError(null)
      return
    }

    let cancelled = false
    setLoading(true)
    setError(null)
    setData(null)

    readHdf5File(source)
      .then((result) => {
        if (!cancelled) {
          setData(result)
          setLoading(false)
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err))
          setLoading(false)
        }
      })

    return () => { cancelled = true }
  }, [source])

  return { data, loading, error }
}
