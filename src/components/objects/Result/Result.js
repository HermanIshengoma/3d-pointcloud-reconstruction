import { Group, MeshBasicMaterial, Mesh } from 'three';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader'


var parentGlobal; 
class Result extends Group {
    
    constructor(parent, url) {
        // Call parent Group() constructor
        super();

        parentGlobal = parent;
        const loader = new PLYLoader();
        this.name = "result";
        loader.load(
        //url.substring(5, url.length)
        url,
        function(geometry){
            //console.log(geometry);
            geometry.computeVertexNormals()
            // https://stackoverflow.com/questions/25735128/three-js-show-single-vertex-as-ie-a-dot
            // https://dev.to/maniflames/pointcloud-effect-in-three-js-3eic
            //const material = new PointsMaterial( { color: 0x808080, size: 1.0/128.0 } )
            //mesh = new Points(geometry, material)
            console.log(geometry);
            const material = new MeshBasicMaterial({color: 0x808080});
            var mesh1 = new Mesh(geometry, material)
            mesh1.rotateX(-Math.PI / 2)
            mesh1.translateX(3)
            
            parent.add(mesh1)
            
            // console.log('Inseide return:', parent.add(mesh1))
        }
        )
        // console.log(parent.getObjectByName('result'));


        
    }

    remove(){
        console.log('Removing....',  parentGlobal.children[2]);
    }
}

export default Result;
