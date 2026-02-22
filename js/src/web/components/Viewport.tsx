import { type ReactNode } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Grid, GizmoHelper, GizmoViewport } from '@react-three/drei'

interface ViewportProps {
  children?: ReactNode
}

export default function Viewport({ children }: ViewportProps) {
  return (
    <Canvas camera={{ position: [1.5, 1.5, 1.5], fov: 50 }}>
      {/* Background — matches PyVista's default dark blue-grey */}
      <color attach="background" args={['#151520']} />

      {/* Lighting */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[5, 8, 5]} intensity={1.2} />

      {/* Navigation */}
      <OrbitControls makeDefault />

      {/* World axes at origin */}
      <axesHelper args={[0.5]} />

      {/* Floor grid at Y = 0 */}
      <Grid
        args={[10, 10]}
        cellSize={0.1}
        cellThickness={0.5}
        cellColor="#3a3a4a"
        sectionSize={0.5}
        sectionThickness={1}
        sectionColor="#5a5a7a"
        fadeDistance={6}
        fadeStrength={1}
        followCamera={false}
        infiniteGrid
      />

      {/* Orientation gizmo — corner indicator matching PyVista's add_axes() */}
      <GizmoHelper alignment="bottom-right" margin={[72, 72]}>
        <GizmoViewport labelColor="white" axisHeadScale={1} />
      </GizmoHelper>

      {children}
    </Canvas>
  )
}
