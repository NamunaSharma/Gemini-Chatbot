// "use client";

// import React, { useEffect, useRef } from "react";
// import * as THREE from "three";
// import { GLTFLoader } from "three/addons/loaders/GLTFLoader";
// import Image from "next/image";
// import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";

// const Page = () => {
//   // Refs
//   const canvasRef = useRef(null);
//   const setup = useRef(null);
//   const cameraRef = useRef(null);
//   const controlsRef = useRef(null);

//   const targetMeshNames = ["Object_5", "Object_4", "Object_7"];
//   const textureLoader = new THREE.TextureLoader();

//   // Tile click
//   const handleClick = (texturePath) => {
//     if (!setup.current) return;

//     const texture = textureLoader.load(texturePath);
//     texture.wrapS = THREE.RepeatWrapping;
//     texture.wrapT = THREE.RepeatWrapping;
//     texture.repeat.set(0.3, 0.3);

//     targetMeshNames.forEach((meshName) => {
//       const mesh = setup.current.getObjectByName(meshName);
//       if (mesh && mesh.isMesh && mesh.material) {
//         mesh.material.map = texture;
//         mesh.material.needsUpdate = true;
//       }
//     });
//   };

//   useEffect(() => {
//     if (!canvasRef.current) return;

//     // Scene
//     const scene = new THREE.Scene();
//     scene.background = null;

//     // Camera
//     const camera = new THREE.PerspectiveCamera(
//       75,
//       window.innerWidth / window.innerHeight,
//       0.1,
//       1000
//     );
//     camera.position.set(5, 3, 6);
//     cameraRef.current = camera;

//     // Renderer
//     const renderer = new THREE.WebGLRenderer({ alpha: true });
//     renderer.setSize(window.innerWidth, window.innerHeight);
//     canvasRef.current.appendChild(renderer.domElement);

//     // Controls
//     const controls = new OrbitControls(camera, renderer.domElement);
//     controls.enableDamping = true;
//     controls.dampingFactor = 0.05;
//     controlsRef.current = controls;

//     // Ambient light
//     const ambientLight = new THREE.AmbientLight("white", 2);
//     scene.add(ambientLight);

//     // Load GLTF model
//     const loader = new GLTFLoader();
//     loader.load("/assets/swimming_pool_343.glb", (gltf) => {
//       const model = gltf.scene;
//       setup.current = model;
//       model.scale.set(0.4, 0.4, 0.4);
//       model.position.set(0, 1, 0);
//       scene.add(model);

//       // Default texture
//       const texture = textureLoader.load("/assets/Tile1.jpg");
//       texture.wrapS = THREE.RepeatWrapping;
//       texture.wrapT = THREE.RepeatWrapping;
//       texture.repeat.set(0.3, 0.3);

//       targetMeshNames.forEach((meshName) => {
//         const mesh = setup.current.getObjectByName(meshName);
//         if (mesh && mesh.isMesh && mesh.material) {
//           mesh.material.map = texture;
//           mesh.material.needsUpdate = true;
//         }
//       });
//     });

//     // Animate
//     const animate = () => {
//       requestAnimationFrame(animate);
//       controls.update();
//       renderer.render(scene, camera);
//     };
//     animate();

//     // Cleanup
//     return () => {
//       if (canvasRef.current) canvasRef.current.removeChild(renderer.domElement);
//     };
//   }, []);

//   // Button handlers
//   const setTopView = () => {
//     if (!cameraRef.current || !controlsRef.current) return;
//     // cameraRef.current.position.set(0, 10, 0);
//     cameraRef.current.position.set(0, 300, 295);

//     controlsRef.current.target.set(0, 0, 0);
//     controlsRef.current.update();
//   };

//   const setSideView = () => {
//     if (!cameraRef.current || !controlsRef.current) return;
//     cameraRef.current.position.set(200, 163, 0);
//     controlsRef.current.target.set(0, 1, 0);
//     controlsRef.current.update();
//   };

//   const setIsometricView = () => {
//     if (!cameraRef.current || !controlsRef.current) return;
//     cameraRef.current.position.set(5, 5, 5);
//     controlsRef.current.target.set(0, 1, 0);
//     controlsRef.current.update();
//   };

//   const rotateRight = () => {
//     if (!cameraRef.current || !controlsRef.current) return;
//     cameraRef.current.position.applyAxisAngle(
//       new THREE.Vector3(0, 1, 0),
//       Math.PI / 6
//     );
//     controlsRef.current.update();
//   };

//   const rotateLeft = () => {
//     if (!cameraRef.current || !controlsRef.current) return;
//     cameraRef.current.position.applyAxisAngle(
//       new THREE.Vector3(0, 1, 0),
//       -Math.PI / 6
//     );
//     controlsRef.current.update();
//   };

//   const zoomIn = () => {
//     if (!cameraRef.current || !controlsRef.current) return;
//     cameraRef.current.position.multiplyScalar(0.9);
//     controlsRef.current.update();
//   };

//   const zoomOut = () => {
//     if (!cameraRef.current || !controlsRef.current) return;
//     cameraRef.current.position.multiplyScalar(1.1);
//     controlsRef.current.update();
//   };

//   return (
//     <div className="grid place-items-center">
//       <h1 className="text-2xl font-bold mb-4">3D Pool Tile Preview</h1>

//       {/* Tile selection */}
//       <div className="flex justify-center gap-4 mb-4">
//         {["Tile1", "picture"].map((tile, i) => (
//           <Image
//             key={i}
//             src={`/assets/${tile}.jpg`}
//             width={100}
//             height={100}
//             alt={tile}
//             className="border-2 cursor-pointer"
//             onClick={() => handleClick(`/assets/${tile}.jpg`)}
//           />
//         ))}
//       </div>

//       {/* View & control buttons */}
//       <div className="flex flex-wrap justify-center gap-2 mb-4">
//         <button
//           className="px-4 py-2 bg-blue-500 text-white rounded"
//           onClick={setTopView}
//         >
//           Top View
//         </button>
//         <button
//           className="px-4 py-2 bg-green-500 text-white rounded"
//           onClick={setSideView}
//         >
//           Side View
//         </button>
//         <button
//           className="px-4 py-2 bg-purple-500 text-white rounded"
//           onClick={setIsometricView}
//         >
//           Isometric
//         </button>
//         <button
//           className="px-4 py-2 bg-yellow-500 text-white rounded"
//           onClick={rotateRight}
//         >
//           Rotate Right
//         </button>
//         <button
//           className="px-4 py-2 bg-yellow-500 text-white rounded"
//           onClick={rotateLeft}
//         >
//           Rotate Left
//         </button>
//         <button
//           className="px-4 py-2 bg-red-500 text-white rounded"
//           onClick={zoomIn}
//         >
//           Zoom In
//         </button>
//         <button
//           className="px-4 py-2 bg-red-500 text-white rounded"
//           onClick={zoomOut}
//         >
//           Zoom Out
//         </button>
//       </div>

//       {/* Canvas */}
//       <div ref={canvasRef} className="w-full h-screen"></div>
//     </div>
//   );
// };

// export default Page;
