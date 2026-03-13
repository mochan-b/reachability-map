import { Suspense, useMemo, useEffect } from 'react'
import * as THREE from 'three'
import { useLoader } from '@react-three/fiber'
import { ColladaLoader } from 'three-stdlib'

const PI = Math.PI
type n = number

// Franka Panda ready/home joint angles from Robotics Toolbox's URDF model:
// q = [0, -0.3, 0, -2.2, 0, 2.0, pi/4]
const PANDA_HOME_Q: [n, n, n, n, n, n, n] = [0, -0.3, 0, -2.2, 0, 2.0, PI / 4]
const PANDA_FINGER_OPENING = 0.04

type JointMotion =
  | { type: 'revolute'; axis: [n, n, n]; value: n }
  | { type: 'prismatic'; axis: [n, n, n]; value: n }

// Joint data extracted from assets/panda.urdf plus franka_description's
// hand.xacro for the finger visuals. Each entry: { xyz, rpy, parent, motion }
// where parent is the index of the parent joint (-1 = world root).
// Indices:
//   0: link0 root
//   1-7: panda_link1..panda_link7 revolute joints
//   8: panda_link8 fixed joint
//   9: panda_hand fixed joint
//   10-11: left/right finger prismatic joints
const JOINTS: { xyz: [n, n, n]; rpy: [n, n, n]; parent: number; motion?: JointMotion }[] = [
  { xyz: [0, 0, 0],           rpy: [0, 0, 0],           parent: -1 }, // 0: link0
  { xyz: [0, 0, 0.333],       rpy: [0, 0, 0],           parent: 0, motion: { type: 'revolute', axis: [0, 0, 1], value: PANDA_HOME_Q[0] } }, // 1
  { xyz: [0, 0, 0],           rpy: [-PI/2, 0, 0],       parent: 1, motion: { type: 'revolute', axis: [0, 0, 1], value: PANDA_HOME_Q[1] } }, // 2
  { xyz: [0, -0.316, 0],      rpy: [PI/2, 0, 0],        parent: 2, motion: { type: 'revolute', axis: [0, 0, 1], value: PANDA_HOME_Q[2] } }, // 3
  { xyz: [0.0825, 0, 0],      rpy: [PI/2, 0, 0],        parent: 3, motion: { type: 'revolute', axis: [0, 0, 1], value: PANDA_HOME_Q[3] } }, // 4
  { xyz: [-0.0825, 0.384, 0], rpy: [-PI/2, 0, 0],       parent: 4, motion: { type: 'revolute', axis: [0, 0, 1], value: PANDA_HOME_Q[4] } }, // 5
  { xyz: [0, 0, 0],           rpy: [PI/2, 0, 0],        parent: 5, motion: { type: 'revolute', axis: [0, 0, 1], value: PANDA_HOME_Q[5] } }, // 6
  { xyz: [0.088, 0, 0],       rpy: [PI/2, 0, 0],        parent: 6, motion: { type: 'revolute', axis: [0, 0, 1], value: PANDA_HOME_Q[6] } }, // 7
  { xyz: [0, 0, 0.107],       rpy: [0, 0, 0],           parent: 7 }, // 8: link8
  { xyz: [0, 0, 0],           rpy: [0, 0, -PI/4],       parent: 8 }, // 9: hand
  { xyz: [0, 0, 0.0584],      rpy: [0, 0, 0],           parent: 9, motion: { type: 'prismatic', axis: [0, 1, 0], value: PANDA_FINGER_OPENING } }, // 10: left finger
  { xyz: [0, 0, 0.0584],      rpy: [0, 0, 0],           parent: 9, motion: { type: 'prismatic', axis: [0, -1, 0], value: PANDA_FINGER_OPENING } }, // 11: right finger
]

const ZERO_XYZ: [n, n, n] = [0, 0, 0]
const ZERO_RPY: [n, n, n] = [0, 0, 0]

const MESH_DEFS: {
  jointIndex: number
  url: string
  visualXyz?: [n, n, n]
  visualRpy?: [n, n, n]
}[] = [
  { jointIndex: 0, url: '/panda_meshes/link0.dae' },
  { jointIndex: 1, url: '/panda_meshes/link1.dae' },
  { jointIndex: 2, url: '/panda_meshes/link2.dae' },
  { jointIndex: 3, url: '/panda_meshes/link3.dae' },
  { jointIndex: 4, url: '/panda_meshes/link4.dae' },
  { jointIndex: 5, url: '/panda_meshes/link5.dae' },
  { jointIndex: 6, url: '/panda_meshes/link6.dae' },
  { jointIndex: 7, url: '/panda_meshes/link7.dae' },
  { jointIndex: 9, url: '/panda_meshes/hand.dae' },
  { jointIndex: 10, url: '/panda_meshes/finger.dae' },
  { jointIndex: 11, url: '/panda_meshes/finger.dae', visualRpy: [0, 0, PI] },
]
const MESH_URLS = MESH_DEFS.map(({ url }) => url)

// Build a 4x4 transform from URDF <origin xyz rpy>, followed by optional joint
// motion in the joint frame.
function jointTransform(
  xyz: [n, n, n],
  rpy: [n, n, n],
  motion?: JointMotion
): THREE.Matrix4 {
  const origin = new THREE.Matrix4()
  origin.makeRotationFromEuler(new THREE.Euler(rpy[0], rpy[1], rpy[2], 'XYZ'))
  origin.setPosition(xyz[0], xyz[1], xyz[2])
  if (!motion || motion.value === 0) return origin

  if (motion.type === 'revolute') {
    const axis = new THREE.Vector3(...motion.axis).normalize()
    return origin.multiply(new THREE.Matrix4().makeRotationAxis(axis, motion.value))
  }

  return origin.multiply(
    new THREE.Matrix4().makeTranslation(
      motion.axis[0] * motion.value,
      motion.axis[1] * motion.value,
      motion.axis[2] * motion.value,
    )
  )
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
  for (const { xyz, rpy, parent, motion } of JOINTS) {
    const T = jointTransform(xyz, rpy, motion)
    if (parent === -1) {
      fkRobot.push(T)
    } else {
      fkRobot.push(fkRobot[parent].clone().multiply(T))
    }
  }
  // Final group matrix: M = F * T_fk * Rx(+π/2)
  return MESH_DEFS.map(({ jointIndex, visualXyz = ZERO_XYZ, visualRpy = ZERO_RPY }) =>
    F.clone()
      .multiply(fkRobot[jointIndex])
      .multiply(jointTransform(visualXyz, visualRpy))
      .multiply(RxHalfPi)
  )
}

const LINK_MATRICES = computeLinkMatrices()

// ──────────────────────────────────────────────────────────────────────────────

interface LinkMeshProps {
  url: string
  matrix: THREE.Matrix4
  opacity: number
}

function cloneMaterial(material: THREE.Material): THREE.Material {
  const clone = material.clone()
  clone.userData = {
    ...clone.userData,
    baseOpacity: material.opacity,
    baseTransparent: material.transparent,
    baseDepthWrite: material.depthWrite,
  }
  return clone
}

function cloneMeshMaterials(material: THREE.Material | THREE.Material[]) {
  return Array.isArray(material)
    ? material.map((entry) => cloneMaterial(entry))
    : cloneMaterial(material)
}

function updateMeshMaterials(
  material: THREE.Material | THREE.Material[],
  opacity: number
): void {
  const materials = Array.isArray(material) ? material : [material]
  for (const mat of materials) {
    const baseOpacity =
      typeof mat.userData.baseOpacity === 'number' ? mat.userData.baseOpacity : 1
    const baseTransparent = mat.userData.baseTransparent === true
    const baseDepthWrite =
      typeof mat.userData.baseDepthWrite === 'boolean' ? mat.userData.baseDepthWrite : true
    const effectiveOpacity = baseOpacity * opacity

    mat.opacity = effectiveOpacity
    mat.transparent = baseTransparent || effectiveOpacity < 0.999
    mat.depthWrite = baseDepthWrite && effectiveOpacity >= 0.999
    mat.needsUpdate = true
  }
}

function LinkMesh({ url, matrix, opacity }: LinkMeshProps) {
  const collada = useLoader(ColladaLoader, url)

  // Clone mesh scene once per loaded file and preserve the embedded COLLADA
  // materials so the Panda's source colors survive in the web view.
  const scene = useMemo(() => {
    const clone = collada.scene.clone(true)
    clone.traverse((obj) => {
      const mesh = obj as THREE.Mesh
      if (!mesh.isMesh) return
      mesh.material = cloneMeshMaterials(mesh.material)
    })
    return clone
  }, [collada])

  // Update opacity in-place to avoid re-cloning on every slider drag.
  useEffect(() => {
    scene.traverse((obj) => {
      const mesh = obj as THREE.Mesh
      if (!mesh.isMesh) return
      updateMeshMaterials(mesh.material, opacity)
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
        <LinkMesh key={`${url}:${i}`} url={url} matrix={LINK_MATRICES[i]} opacity={opacity} />
      ))}
    </Suspense>
  )
}
