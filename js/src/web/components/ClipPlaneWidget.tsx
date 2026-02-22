import { useCallback } from 'react'
import { PivotControls } from '@react-three/drei'
import * as THREE from 'three'

interface Props {
  onPlaneChange: (plane: THREE.Plane) => void
}

// Scratch vectors — reused across drag events
const _n = new THREE.Vector3()
const _p = new THREE.Vector3()

export default function ClipPlaneWidget({ onPlaneChange }: Props) {
  const handleDrag = useCallback(
    (_l: THREE.Matrix4, _dl: THREE.Matrix4, w: THREE.Matrix4) => {
      // Disc lies in the local XZ plane, so the plane normal is local +Y.
      // Column 1 of the world matrix = world-space direction of local +Y.
      _n.set(w.elements[4], w.elements[5], w.elements[6]).normalize()
      _p.setFromMatrixPosition(w)
      onPlaneChange(new THREE.Plane().setFromNormalAndCoplanarPoint(_n, _p))
    },
    [onPlaneChange],
  )

  return (
    <PivotControls scale={0.7} lineWidth={2} onDrag={handleDrag}>
      {/* Semi-transparent disc — depthTest:false keeps it visible through voxels */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[1.2, 64]} />
        <meshBasicMaterial
          color="#22d3ee"
          transparent
          opacity={0.18}
          side={THREE.DoubleSide}
          depthTest={false}
          depthWrite={false}
        />
      </mesh>

      {/* Solid border ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[1.16, 1.24, 64]} />
        <meshBasicMaterial
          color="#22d3ee"
          side={THREE.DoubleSide}
          depthTest={false}
          depthWrite={false}
        />
      </mesh>

      {/* Cross lines on the disc surface for orientation reference */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0, 0.03, 8]} />
        <meshBasicMaterial color="#22d3ee" depthTest={false} depthWrite={false} />
      </mesh>
      <mesh position={[0, 0, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[2.4, 0.025]} />
        <meshBasicMaterial
          color="#22d3ee"
          transparent
          opacity={0.5}
          side={THREE.DoubleSide}
          depthTest={false}
          depthWrite={false}
        />
      </mesh>
      <mesh position={[0, 0, 0]} rotation={[-Math.PI / 2, Math.PI / 2, 0]}>
        <planeGeometry args={[2.4, 0.025]} />
        <meshBasicMaterial
          color="#22d3ee"
          transparent
          opacity={0.5}
          side={THREE.DoubleSide}
          depthTest={false}
          depthWrite={false}
        />
      </mesh>

      {/* Normal indicator arrow — points in local +Y (the plane normal direction) */}
      <mesh position={[0, 0.3, 0]}>
        <cylinderGeometry args={[0.018, 0.018, 0.4, 8]} />
        <meshBasicMaterial color="#22d3ee" depthTest={false} depthWrite={false} />
      </mesh>
      <mesh position={[0, 0.57, 0]}>
        <coneGeometry args={[0.055, 0.14, 8]} />
        <meshBasicMaterial color="#22d3ee" depthTest={false} depthWrite={false} />
      </mesh>
    </PivotControls>
  )
}
