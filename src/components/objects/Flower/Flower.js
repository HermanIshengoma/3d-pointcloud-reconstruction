import { Group, MeshBasicMaterial, Mesh, Points, PointsMaterial, Vector3, Geometry, Euler  } from 'three';
import { TWEEN } from 'three/examples/jsm/libs/tween.module.min.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader'
import {Result} from 'objects';

var mesh = null;
var geo;
var parentState;
var parentGlobal;
var res;
class Flower extends Group {

    constructor(parent, meshObj, pState, folders) {
        // Call parent Group() constructor
        super();

        parentGlobal = parent;
        parentState = pState;

        
        // Init state
        this.state = {
            gui: parent.state.gui,
            bob: true,
            pointCloudRand: this.pointCloudRand.bind(this),
            resolution: 96,
            twirl: 0,
            noiseApply: this.noise.bind(this),
            inflateApply: this.inflate.bind(this),
            twistApply: this.twist.bind(this),
            factorNoise: 1.0/8.0,
            factorInflate: 1.0/8.0,
            factorTwist: 1.0
            
        };

        
        // Load object
        const loader = new PLYLoader();

        this.name = 'baseModel';
        const path = './src/components/objects/models/' + meshObj;

        async function fetchMesh() {
            const response = await fetch('https://final-3d-reconstruction.herokuapp.com/post/', {
          //const response = await fetch('http://127.0.0.1:5000/post/', {
            method: 'POST',
            headers: {
              //'Accept': 'application/json',
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({'data': JSON.stringify([[]])})
          });
          const content = await response.blob();
          return content;
        }
        fetchMesh().then(content => {
            //console.log(content); // fetched movies
            var url = URL.createObjectURL(content);
             // https://sbcode.net/threejs/loaders-ply/
            loader.load(
                url,
                function (geometry) {
                    geometry.computeVertexNormals()
                    // https://stackoverflow.com/questions/25735128/three-js-show-single-vertex-as-ie-a-dot
                    // https://dev.to/maniflames/pointcloud-effect-in-three-js-3eic
                    //const material = new PointsMaterial( { color: 0x808080, size: 1.0/128.0 } )
                    //mesh = new Points(geometry, material)
                    //const material = new MeshBasicMaterial({color: 0x808080});
                    // const material = new MeshBasicMaterial({color: 0xf0f0f0});
                    const material = new MeshBasicMaterial({color: 0x5DADE2 });
                    mesh = new Mesh(geometry, material)
                    mesh.rotateX(-Math.PI / 2)
            
                    // ensuring mesh is inside unit cube or encompasses most of it
                    geometry.computeBoundingBox();
                    // console.log('before bounding box', geometry.boundingBox)
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
            
                    geo = geometry;
                    parent.add(mesh)
            
                },
                (xhr) => {
                    console.log((xhr.loaded / xhr.total) * 100 + '% loaded')
                },
                (error) => {
                    console.log(error)
                }
            )
            });

        // Add self to parent's update list
        parent.addToUpdateList(this);

        // Populate GUI
        // this.state.gui.add(this.state, 'bob');
        folders['PointCloudGeneration'].add(this.state, 'resolution', 64, 128);
        folders['PointCloudGeneration'].add(this.state, 'pointCloudRand');

        const filter = this.state.gui.addFolder('Filters');
        var fold = filter.addFolder('noise')
        fold.add(this.state, 'factorNoise', 1/128.0, 1/8.0)
        fold.add(this.state, 'noiseApply');

        var fold = filter.addFolder('inflate')
        fold.add(this.state, 'factorInflate', 1/128.0, 1/8.0)
        fold.add(this.state, 'inflateApply');

        var fold = filter.addFolder('twist')
        fold.add(this.state, 'factorTwist', 0.0, 5.0);
        fold.add(this.state, 'twistApply');
    }

    // randomly select random set of vertices
    pointCloudRand() {
        if(parentGlobal.children.length == 4){
            // console.log(parentGlobal.children[3]['uuid'])
            const object = parentGlobal.getObjectByProperty( 'uuid', parentGlobal.children[3]['uuid']);
            // referencing https://discourse.threejs.org/t/correctly-remove-mesh-from-scene-and-dispose-material-and-geometry/5448/2
            object.geometry.dispose();
            object.material.dispose();
            parentGlobal.remove( object );
            
            // console.log(parentGlobal.children);
        }
        const pos = geo.attributes.position;
        const numVertices = pos.count;
        const samples = parentState.numSamples;
        var pointCloud = [];
        var tracker = []
        var resolution = Math.round(this.state.resolution);
        

        for (var i = 0; i < samples; i++){
            var rand = Math.floor(Math.random() * numVertices);
            while(tracker.includes(rand)) rand = Math.floor(Math.random() * numVertices);
            pointCloud.push([pos.array[rand * 3], pos.array[(rand * 3) + 1], pos.array[(rand * 3) + 2]]);
            tracker.push(rand);
        }
        console.log(pointCloud);
        // console.log(geo.attributes.position)
        // send request
        async function fetchMesh() {
            const response = await fetch('https://final-3d-reconstruction.herokuapp.com/post/', {
          //const response = await fetch('http://127.0.0.1:5000/post/', {
            method: 'POST',
            headers: {
              //'Accept': 'application/json',
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({'data': JSON.stringify(pointCloud)})
            //body: JSON.stringify({'data': pointCloud}),
          });
          const content = await response.blob();
          return content;
        }
        
        fetchMesh().then(content => {
          //console.log(content); // fetched movies
          var url = URL.createObjectURL(content);
          res = new Result(parentGlobal, url)
        });

    }

    noise(){
        var numVertices = geo.attributes.position.count;
        // obtained idea of making new geometry from https://stackoverflow.com/questions/61269336/find-neigbouring-vertices-of-a-vertex
        var geoProper = new Geometry().fromBufferGeometry( geo );

        var edge_lens = []
        for (var i = 0; i < numVertices; i++) {
            edge_lens.push(this.averageEdgeLengths(i, geoProper));
        }

        var factor = this.state.factorNoise;
        for (var i = 0; i < numVertices; i++) {
            // max min adapted from https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/random
            var rand = Math.random() * (1 - -1)+ -1;
            let toAdd = (this.getNormal(i)).multiplyScalar(rand * factor * edge_lens[i]);
            var updatedVertex = this.getVertex(i).add(toAdd);
            this.updateVertex(i, updatedVertex);
        }
        this.boundMesh();
       
    }
    
    inflate(){
        var numVertices = geo.attributes.position.count;
        // obtained idea of making new geometry from https://stackoverflow.com/questions/61269336/find-neigbouring-vertices-of-a-vertex
        var geoProper = new Geometry().fromBufferGeometry( geo );

        var edge_lens = []
        for (var i = 0; i < numVertices; i++) {
            edge_lens.push(this.averageEdgeLengths(i, geoProper));
            // edge_lens.push(1)
        }

        var factor = this.state.factorInflate;
        for (var i = 0; i < numVertices; i++) {            
            let toAdd = (this.getNormal(i)).multiplyScalar(factor * edge_lens[i]);
            var updatedVertex = this.getVertex(i).add(toAdd);
            this.updateVertex(i, updatedVertex);
        }
        
        this.boundMesh();
        

    }

    twist(){
        // console.log(' bounding box', geo.boundingBox)
        var numVertices = geo.attributes.position.count;
        var factor = this.state.factorTwist;
        for (var i = 0; i < numVertices; i++) {
            var v = this.getVertex(i);
            var a = new Euler(0, v.getComponent(1) * factor, 0, 'XYZ');
            v.applyEuler(a);
            this.updateVertex(i, v);
        }
        this.boundMesh();

    }

    update(timeStamp) {
        if (this.state.bob) {
            // Bob back and forth
            this.rotation.z = 0.05 * Math.sin(timeStamp / 300);
        }
        if (this.state.twirl > 0) {
            // Lazy implementation of twirl
            this.state.twirl -= Math.PI / 8;
            this.rotation.y += Math.PI / 8;
        }

        // Advance tween animations, if any exist
        TWEEN.update();
    }

    // mesh functions
    // get vertex of mesh at index i
    getVertex(i){
        const pos = geo.attributes.position;
        var x = i*3;
        return new Vector3(pos.array[x], pos.array[x+1], pos.array[x+2]);
    }

    updateVertex(i, vert){
        const pos = geo.attributes.position;
        var x = i*3;
        pos.array[x] = vert.x;
        pos.array[x+1] = vert.y;
        pos.array[x+2] = vert.z;
        // reference for line below: https://stackoverflow.com/questions/24531109/three-js-vertices-does-not-update
        geo.attributes.position.needsUpdate = true;

    }

    getNormal(i){
        const pos = geo.attributes.normal;
        var x = i*3;
        return new Vector3(pos.array[x], pos.array[x+1], pos.array[x+2]);
    }

    updateNormal(i, vert){
        const pos = geo.attributes.normal;
        var x = i*3;
        pos.array[x] = vert.x;
        pos.array[x+1] = vert.y;
        pos.array[x+2] = vert.z;
        // reference for line below: https://stackoverflow.com/questions/24531109/three-js-vertices-does-not-update
        geo.attributes.normal.needsUpdate = true;

    }

    //finds the neighbouring vertices of the given vertex
    neighbouringVerts(vertexIndex, geoProper){
        var faceNum = geoProper.faces.length;
        var neighbours = new Set();

        for(var i = 0; i < faceNum; i++){
            if((geoProper.faces[i].a == vertexIndex || geoProper.faces[i].b == vertexIndex ) || geoProper.faces[i].c == vertexIndex){
                if(geoProper.faces[i].a != vertexIndex ) neighbours.add(geoProper.faces[i].a);
                if(geoProper.faces[i].b != vertexIndex ) neighbours.add(geoProper.faces[i].b);
                if(geoProper.faces[i].c != vertexIndex ) neighbours.add(geoProper.faces[i].c);
                
            }
        }
        
        return neighbours;
    }

    averageEdgeLengths(vertexIndex, geoProper){
    var avg = 0.0;
    var v = this.getVertex(vertexIndex);
    var neighbours = this.neighbouringVerts(vertexIndex, geoProper);
    for (let i = 0; i< neighbours.size; i++){
        var neighbourPos = this.getVertex(neighbours[i]);
        let edge_len = (neighbourPos).distanceTo(v);
        // console.log(edge_len);
        avg += edge_len;
    }

    return avg / neighbours.size;
    }

    // ensuring mesh is inside unit cube or encompasses most of it
    async boundMesh(){
        geo.computeBoundingBox();
        // console.log('before bounding box', geo.boundingBox)
        var max = 0.0;
        if (Math.abs(geo.boundingBox.max.x) > max) max = geo.boundingBox.max.x
        if (Math.abs(geo.boundingBox.max.y) > max) max = geo.boundingBox.max.y
        if (Math.abs(geo.boundingBox.max.z) > max) max = geo.boundingBox.max.z
        if (Math.abs(geo.boundingBox.min.x) > max) max = geo.boundingBox.min.x
        if (Math.abs(geo.boundingBox.min.y) > max) max = geo.boundingBox.min.y
        if (Math.abs(geo.boundingBox.min.z) > max) max = geo.boundingBox.min.z
        max = Math.abs(max);
        var scale = 1.0/max;
        geo.scale(scale, scale, scale);
        geo.computeBoundingBox();
        // console.log('after bounding box', geo.boundingBox)

        
    }
}



export default Flower;
