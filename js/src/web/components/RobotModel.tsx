import { Suspense, useMemo, useEffect } from 'react'
import * as THREE from 'three'
import { useLoader } from '@react-three/fiber'
import { ColladaLoader } from 'three-stdlib'

const PI = Math.PI

// Joint data extracted from assets/panda.urdf
// Each entry: { xyz, rpy, parent } where parent is the index of the parent joint (-1 = world root)
// Indices 0–8 correspond to joints panda_joint1 through fixed_joint (link8)
// Index 9 is the panda_hand joint connecting link8 → hand
type n = number
const JOINTS: { xyz: [n, n, n]; rpy: [n, n, n]; parent: number }[] = [
  { xyz: [0, 0, 0],           rpy: [0, 0, 0],           parent: -1 }, // 0: link0 (root)
  { xyz: [0, 0, 0.333],       rpy: [0, 0, 0],           parent: 0  }, // 1: link1
  { xyz: [0, 0, 0],           rpy: [-PI/2, 0, 0],       parent: 1  }, // 2: link2
  { xyz: [0, -0.316, 0],      rpy: [PI/2, 0, 0],        parent: 2  }, // 3: link3
  { xyz: [0.0825, 0, 0],      rpy: [PI/2, 0, 0],        parent: 3  }, // 4: link4
  { xyz: [-0.0825, 0.384, 0], rpy: [-PI/2, 0, 0],       parent: 4  }, // 5: link5
  { xyz: [0, 0, 0],           rpy: [PI/2, 0, 0],        parent: 5  }, // 6: link6
  { xyz: [0.088, 0, 0],       rpy: [PI/2, 0, 0],        parent: 6  }, // 7: link7
  { xyz: [0, 0, 0.107],       rpy: [0, 0, 0],           parent: 7  }, // 8: link8 (fixed, no mesh)
  { xyz: [0, 0, 0],           rpy: [0, 0, -PI/4],       parent: 8  }, // 9: hand joint
]

// Which joint indices have meshes, and what URL to load
const MESH_LINK_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 9]
const MESH_URLS = MESH_LINK_INDICES.map((i) =>
  i < 8 ? `/panda_meshes/link${i}.dae` : '/panda_meshes/hand.dae'
)

// Build a 4×4 transform from URDF <origin xyz rpy>:
// T = [R(rpy) | xyz; 0 0 0 1]  (rotate-then-translate in parent frame)
function jointTransform(xyz: [n, n, n], rpy: [n, n, n]): THREE.Matrix4 {
  const m = new THREE.Matrix4()
  m.makeRotationFromEuler(new THREE.Euler(rpy[0], rpy[1], rpy[2], 'XYZ'))
  m.setPosition(xyz[0], xyz[1], xyz[2])
  return m
}

// Coordinate reflection: robot(x,y,z) → three(x,z,y)
// Swaps Y and Z axes to go from robot frame to Three.js world frame.
const F = new THREE.Matrix4().set(
  1, 0, 0, 0,
  0, 0, 1, 0,   // three.y = robot.z
  0, 1, 0, 0,   // three.z = robot.y
  0, 0, 0, 1,
)

// ColladaLoader applies Rx(-π/2) to every loaded scene to convert Z-up → Y-up.
// To counteract this and get the correct final pose, we right-multiply by Rx(+π/2).
const RxHalfPi = new THREE.Matrix4().makeRotationX(Math.PI / 2)

// Compute world-space FK matrices for each link (robot frame), then convert to
// Three.js frame. Called once at module load — no React state involved.
function computeLinkMatrices(): THREE.Matrix4[] {
  const fkRobot: THREE.Matrix4[] = []
  for (const { xyz, rpy, parent } of JOINTS) {
    const T = jointTransform(xyz, rpy)
    if (parent === -1) {
      fkRobot.push(T)
    } else {
      fkRobot.push(fkRobot[parent].clone().multiply(T))
    }
  }
  // Final group matrix: M = F * T_fk * Rx(+π/2)
  return MESH_LINK_INDICES.map((i) =>
    F.clone().multiply(fkRobot[i]).multiply(RxHalfPi)
  )
}

const LINK_MATRICES = computeLinkMatrices()

// ──────────────────────────────────────────────────────────────────────────────

interface LinkMeshProps {
  url: string
  matrix: THREE.Matrix4
  opacity: number
}

function LinkMesh({ url, matrix, opacity }: LinkMeshProps) {
  const collada = useLoader(ColladaLoader, url)

  // Clone mesh scene once per loaded file; replace materials with grey standard
  const scene = useMemo(() => {
    const clone = collada.scene.clone(true)
    clone.traverse((obj) => {
      const mesh = obj as THREE.Mesh
      if (!mesh.isMesh) return
      mesh.material = new THREE.MeshStandardMaterial({
        color: '#888888',
        transparent: true,
        opacity: 1,
        depthWrite: false,
      })
    })
    return clone
  }, [collada])

  // Update opacity in-place to avoid re-cloning on every slider drag
  useEffect(() => {
    scene.traverse((obj) => {
      const mesh = obj as THREE.Mesh
      if (!mesh.isMesh) return
      const mat = mesh.material as THREE.MeshStandardMaterial
      mat.opacity = opacity
      mat.transparent = opacity < 0.999
      mat.needsUpdate = true
    })
  }, [scene, opacity])

  return (
    <group matrixAutoUpdate={false} matrix={matrix}>
      <primitive object={scene} />
    </group>
  )
}

// ──────────────────────────────────────────────────────────────────────────────

interface RobotModelProps {
  visible: boolean
  opacity: number
}

export default function RobotModel({ visible, opacity }: RobotModelProps) {
  if (!visible) return null
  return (
    <Suspense fallback={null}>
      {MESH_URLS.map((url, i) => (
        <LinkMesh key={url} url={url} matrix={LINK_MATRICES[i]} opacity={opacity} />
      ))}
    </Suspense>
  )
}
