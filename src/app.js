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
import React from 'react'
import ReactDOM from 'react-dom'; 
// import render from 'react-dom' 
import Popup from 'reactjs-popup';
//import swal from 'sweetalert';
import Swal from 'sweetalert2';



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

// swal("A wild Pikachu appeared! What do you want to do?", {
//     buttons: {
//       cancel: "Run away!",
//       catch: {
//         text: "Throw Pokéball!",
//         value: "catch",
//       },
//       defeat: true,
//     },
//   })
//   .then((value) => {
//     switch (value) {
   
//       case "defeat":
//         swal("Pikachu fainted! You gained 500 XP!");
//         break;
   
//       case "catch":
//         swal("Gotcha!", "Pikachu was caught!", "success");
//         break;
   
//       default:
//         swal("Got away safely!");
//     }
//   });

// swal({
//     title: "Welcome to \'Reconstructing 3D Point Clouds\'! ",
//     text: "This is an interactie web app for generating meshes from point clouds. Press next to continue with the tutorial.",
//     buttons: {
//         cancel: "Cancel",
//         next: {
//           value: "next",
//         }
//       },

//   }).then((value) => {
//     switch (value) {
//       case "next":
//         swal("Pikachu fainted! You gained 500 XP!");
//         break;
//     }
//   });

// Swal.fire({
//     title: 'Welcome to \'Reconstructing 3D Point Clouds\'!',
//     text: 'This is an interactie web app for generating meshes from point clouds. Press next to continue with the tutorial.',
//     confirmButtonText: 'Next',
//     showCancelButton: true
//   })

Swal.fire({
    title: 'Welcome to \'Reconstructing 3D Point Clouds\'!',
    text: 'This is an interactie web app for generating meshes from point clouds. You can use the GUI on the top right corner to apply filters to the mesh under the \'Filters\' folder, and you can create a randomly sampled point cloud and return a generated mesh via the \'PointCloudGeneration\' folder. Press \'Next\' to continue with tutorial',
    //showDenyButton: true,
    showCancelButton: true,
    confirmButtonText: `Next`,
    //denyButtonText: `Don't save`,
  }).then((result) => {
    if (result.isConfirmed) {
      //Swal.fire('Saved!', '', 'success')
      scene.state['gui'].__folders['Filters'].open()
      Swal.fire({
          title: "Filters tutorial",
          text: "On the upper right corner we have opened the \'Filters\' folder. There are 5 filters that you can apply to the mesh: noise, inflate, twist, smooth, and sharpen.",
          showCancelButton: true,
          confirmButtonText: `Next`
      }).then((result) => {
        if (result.isConfirmed) {
        scene.state['gui'].__folders['Filters'].__folders['smooth'].open()
        Swal.fire({
            title: "Filters tutorial",
            text: "The way you apply each filter is the same for all. Here we have opened the smooth filter. You can see that it has 3 fields: \'iter\', \'delta\', \'smoothApply\'. You can change the values of iter and delta accordingly and once you have the desired values, just press '\smoothApply\' to apply the filter. It\'s that simple.",
            showCancelButton: true,
            confirmButtonText: `Next`
        }).then((result) => {
            if (result.isConfirmed) {
            scene.state['gui'].__folders['Filters'].__folders['smooth'].close()
            scene.state['gui'].__folders['Filters'].close()
            scene.state['gui'].__folders['PointCloudGeneration'].open()
            Swal.fire({
                title: "Point Cloud tutorial",
                text: "We have now opened the \'PointCloudGeneration\' folder. Here you can edit the fields \'numSamples\' to pick how many pointClouds the program ought to make and \'resolution\' to pick on what resolution the resultant mesh ought to be. Once you have the desired values set, just press pointCloudRand to generate your mesh.",
                confirmButtonText: `Next`
            }).then((result) => {
                scene.state['gui'].__folders['PointCloudGeneration'].close()
                Swal.fire(
                    'Good Job!',
                    'Tutorial Complete! You can now go on and experiment and create your own meshes.',
                    'success'
                  )
            })
        }
        else{
            scene.state['gui'].__folders['Filters'].__folders['smooth'].close()
            scene.state['gui'].__folders['Filters'].close()
        }
        })
    }
        else{
            scene.state['gui'].__folders['Filters'].close()
        }
      })
    }
  })

//console.log(scene.state['gui'].__folders['Filters'].open())
