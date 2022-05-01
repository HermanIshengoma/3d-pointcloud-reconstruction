/**
 * app.js
 *
 * This is the first file loaded. It sets up the Renderer,
 * Scene and Camera. It also starts the render loop and
 * handles window resizes.
 *
 */
import { WebGLRenderer, PerspectiveCamera, Vector3 } from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { SeedScene } from 'scenes';

// Initialize core ThreeJS components
const scene = new SeedScene('reconstruct.ply');
//const scene2 = new SeedScene();
const camera = new PerspectiveCamera();
const renderer = new WebGLRenderer({ antialias: true });

// Set up camera
camera.position.set(6, 3, -10);
camera.lookAt(new Vector3(0, 0, 0));

// Set up renderer, canvas, and minor CSS adjustments
renderer.setPixelRatio(window.devicePixelRatio);
const canvas = renderer.domElement;
canvas.style.display = 'block'; // Removes padding below canvas
document.body.style.margin = 0; // Removes margin around page
document.body.style.overflow = 'hidden'; // Fix scrolling
document.body.appendChild(canvas);

// Set up controls
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.enablePan = false;
controls.minDistance = 4;
controls.maxDistance = 16;
controls.update();

// const fs = require('browserify-fs')
// console.log(`Filename is ${__filename}`);
// const x = `${__filename}`;
// console.log(`Filename is ` + x);
// console.log(`Directory name is ${__dirname}`);
// var d = fs.readFile(x, 'utf-8', (err) => {
//     // In case of a error throw err.
//     if (err) throw err;
// })
// const createServer = require("fs-remote/createServer");

// createServer returns a net.Server
// const server = createServer();

// server.listen(8080, () => {
//   console.log("fs-remote server is listening on port 3000");
// });
// const createClient = fsRemote.createClient;
// const fs = createClient("http://localhost:8080");
// console.log(fs.readFileSync("./package.json"));


// Render loop
const onAnimationFrameHandler = (timeStamp) => {
    controls.update();
    renderer.render(scene, camera);
    //renderer.render(scene2, camera);
    scene.update && scene.update(timeStamp);
    //scene2.update && scene2.update(timeStamp);
    window.requestAnimationFrame(onAnimationFrameHandler);
};
window.requestAnimationFrame(onAnimationFrameHandler);

// Resize Handler
const windowResizeHandler = () => {
    const { innerHeight, innerWidth } = window;
    renderer.setSize(innerWidth, innerHeight);
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
};
windowResizeHandler();
window.addEventListener('resize', windowResizeHandler, false);

// // referencing https://stackoverflow.com/questions/30860773/how-to-get-the-mouse-position-using-three-js
// const mouseLocation = (event) => {
//     var mouse = new Vector3();
//     // mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
//     // mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
//     mouse.x = event.clientX;
//     mouse.y = event.clientY;
//     //mouse.z = event.clientZ;
//     console.log("mouse position: (" + mouse.x + ", "+ mouse.y + ")");
// };
// document.addEventListener("click", mouseLocation, false);


