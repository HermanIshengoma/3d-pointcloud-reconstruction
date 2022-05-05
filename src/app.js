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
import Swal from 'sweetalert2';



// Initialize core ThreeJS components
const scene = new SeedScene('reconstruct.ply');
//const scene2 = new SeedScene();
const camera = new PerspectiveCamera();
const renderer = new WebGLRenderer({ antialias: true });
// const renderer2 = new WebGLRenderer({ antialias: true });

// Set up camera
camera.position.set(6, 3, -10);
camera.lookAt(new Vector3(0, 0, 0));

// Set up renderer, canvas, and minor CSS adjustments
renderer.setPixelRatio(window.devicePixelRatio);
// renderer2.setPixelRatio(window.devicePixelRatio);
// const canvas2 = renderer2.domElement;
const canvas = renderer.domElement;


canvas.style.display = 'block'; // Removes padding below canvas
// canvas2.style.display = 'block';
document.body.style.margin = 0; // Removes margin around page
document.body.style.overflow = 'hidden'; // Fix scrolling
//console.log('Before appendending', document)

document.body.appendChild(canvas);
// document.body.appendChild(canvas2);


//console.log('After appending', document)

// const { innerHeight2, innerWidth2 } = window;
// canvas2.style.left = innerWidth2/2;
// canvas2.style.top = 0;
// //canvas2.offsetLeft = innerWidth2/2;
// console.log('Accessing canvas2:', canvas2)

// Set up controls
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.enablePan = false;
controls.minDistance = 0;
controls.maxDistance = 16;
controls.update();


// Render loop
const onAnimationFrameHandler = (timeStamp) => {
    controls.update();
    // if(scene.state.reconstructed) camera.lookAt(new Vector3(1.5, 0, 0))
    renderer.render(scene, camera);
    // renderer2.render(scene, camera);
    scene.update && scene.update(timeStamp);
    //scene2.update && scene2.update(timeStamp);
    window.requestAnimationFrame(onAnimationFrameHandler);
};
window.requestAnimationFrame(onAnimationFrameHandler);

// Resize Handler
const windowResizeHandler = () => {
    const { innerHeight, innerWidth } = window;
    // renderer2.setSize(innerWidth/2, innerHeight);
    renderer.setSize(innerWidth, innerHeight);
    // canvas2.style.position = 'absolute';
    // canvas2.style.top = innerHeight / 2 - canvas2.height / 2 + 'px';
    // canvas2.style.left = innerWidth / 2 + 'px';
    
    camera.aspect = (innerWidth) / innerHeight;
    camera.updateProjectionMatrix();
};
windowResizeHandler();
window.addEventListener('resize', windowResizeHandler, false);
// Swal.showLoading()
Swal.fire({
    title: 'Welcome to \'Reconstructing 3D Point Clouds\'!',
    text: 'This is an interactie web app for generating meshes from point clouds. You can use the GUI on the top right corner to apply filters to the mesh under the \'Filters\' folder, and you can create a randomly sampled point cloud and return a generated mesh via the \'PointCloudGeneration\' folder. Press \'Next\' to continue with tutorial',
    
    showCancelButton: true,
    confirmButtonText: `Next`,
    //denyButtonText: `Don't save`,
  }).then((result) => {
    if (result.isConfirmed) {
      //Swal.fire('Saved!', '', 'success')
      // Swal.showLoading()
      scene.state['gui'].__folders['Filters'].open()
      Swal.fire({
          title: "Filters Guide",
          text: "On the upper right corner we have opened the \'Filters\' folder. There are 5 filters that you can apply to the mesh: noise, inflate, twist, smooth, and sharpen.",
          showCancelButton: true,
          // showLoaderOnConfirm: true,
          confirmButtonText: `Next`
      }).then((result) => {
        if (result.isConfirmed) {
        scene.state['gui'].__folders['Filters'].__folders['smooth'].open()
        Swal.fire({
            title: "Filters Guide",
            text: "The way you apply each filter is the same for all. Here we have opened the smooth filter. You can see that it has 3 fields: \'iter\', \'delta\', \'smoothApply\'. You can change the values of iter and delta accordingly and once you have the desired values, just press '\smoothApply\' to apply the filter. It\'s that simple.",
            showCancelButton: true,
            confirmButtonText: `Next`
        }).then((result) => {
            if (result.isConfirmed) {
            scene.state['gui'].__folders['Filters'].__folders['smooth'].close()
            scene.state['gui'].__folders['Filters'].close()
            scene.state['gui'].__folders['PointCloudGeneration'].open()
            Swal.fire({
                title: "Point Cloud Guide",
                text: "We have now opened the \'PointCloudGeneration\' folder. Here you can edit the fields \'numSamples\' to pick how many pointClouds the program ought to make and \'resolution\' to pick on what resolution the resultant mesh ought to be. Once you have the desired values set, just press pointCloudRand to generate your mesh.",
                confirmButtonText: `Next`
            }).then((result) => {
                scene.state['gui'].__folders['PointCloudGeneration'].close()
                Swal.fire(
                    'Good Job!',
                    'Guide Complete! You can now go on and experiment and create your own meshes.',
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
