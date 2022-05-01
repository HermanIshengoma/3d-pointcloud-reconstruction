import { Group } from 'three';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader'
import { TWEEN } from 'three/examples/jsm/libs/tween.module.min.js';


class Models extends Group {
    constructor(parent) {
        // Call parent Group() constructor
        super();

        // Init state
        this.state = {
            gui: parent.state.gui,
        };

        // Load object
        const loader = new PLYLoader();

        this.name = 'models';
        loader.load(
            './src/components/objects/Models/reconstruct.ply',
            function (geometry) {
                geometry.computeVertexNormals()
                const material = new MeshBasicMaterial( { color: 0x808080 } )
                const mesh = new Mesh(geometry, material)
                mesh.rotateX(-Math.PI / 2)
        
                // ensuring mesh is inside unit cube or encompasses most of it
                geometry.computeBoundingBox();
                console.log('before bounding box', geometry.boundingBox)
                var max = 0.0;
                if (Math.abs(geometry.boundingBox.max.x) > max) max = geometry.boundingBox.max.x
                if (Math.abs(geometry.boundingBox.max.y) > max) max = geometry.boundingBox.max.y
                if (Math.abs(geometry.boundingBox.max.z) > max) max = geometry.boundingBox.max.z
                if (Math.abs(geometry.boundingBox.min.x) > max) max = geometry.boundingBox.min.x
                if (Math.abs(geometry.boundingBox.min.y) > max) max = geometry.boundingBox.min.y
                if (Math.abs(geometry.boundingBox.min.z) > max) max = geometry.boundingBox.min.z
                max = Math.abs(max);
                var scale = 1.0/max;
                geometry.scale(scale, scale, scale);
                geometry.computeBoundingBox();
        
                scene.add(mesh)
        
                
                // console.log(geometry)
                // fs.writeFile('geometryPos.txt', geometry.attributes.position, (err) => {
                //     // In case of a error throw err.
                //     if (err) throw err;
                // })
            },
            (xhr) => {
                console.log((xhr.loaded / xhr.total) * 100 + '% loaded')
            },
            (error) => {
                console.log(error)
            }
        )

        // Add self to parent's update list
        parent.addToUpdateList(this);

        // Populate GUI
        // this.state.gui.add(this.state, 'bob');
        // this.state.gui.add(this.state, 'spin');
    }

    // spin() {
    //     // Add a simple twirl
    //     this.state.twirl += 6 * Math.PI;

    //     // Use timing library for more precice "bounce" animation
    //     // TweenJS guide: http://learningthreejs.com/blog/2011/08/17/tweenjs-for-smooth-animation/
    //     // Possible easings: http://sole.github.io/tween.js/examples/03_graphs.html
    //     const jumpUp = new TWEEN.Tween(this.position)
    //         .to({ y: this.position.y + 1 }, 300)
    //         .easing(TWEEN.Easing.Quadratic.Out);
    //     const fallDown = new TWEEN.Tween(this.position)
    //         .to({ y: 0 }, 300)
    //         .easing(TWEEN.Easing.Quadratic.In);

    //     // Fall down after jumping up
    //     jumpUp.onComplete(() => fallDown.start());

    //     // Start animation
    //     jumpUp.start();
    // }

    update(timeStamp) {
        // if (this.state.bob) {
        //     // Bob back and forth
        //     this.rotation.z = 0.05 * Math.sin(timeStamp / 300);
        // }
        // if (this.state.twirl > 0) {
        //     // Lazy implementation of twirl
        //     this.state.twirl -= Math.PI / 8;
        //     this.rotation.y += Math.PI / 8;
        // }

        // Advance tween animations, if any exist
        TWEEN.update();
    }
}

export default Models;
