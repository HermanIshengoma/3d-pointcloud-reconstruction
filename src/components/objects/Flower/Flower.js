import { Group, MeshBasicMaterial, Mesh, Points, PointsMaterial, Vector3, Geometry, Euler, MeshStandardMaterial  } from 'three';
import { TWEEN } from 'three/examples/jsm/libs/tween.module.min.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader'
import {Result} from 'objects';
import Swal from 'sweetalert2';
import MODEL from './reconstruct.ply'
import TYPE11 from './bed1.ply'
import TYPE21 from './bed1.ply'
import TYPE22 from './bed2.ply'
import TYPE23 from './bed3.ply'
import TYPE24 from './bed4.ply'
import TYPE31 from './obj1.ply'
import TYPE32 from './obj2.ply'
import TYPE33 from './obj3.ply'
import ANS11 from './bed1_recon_model1.ply'
import ANS21 from './model21ans.ply'
import ANS22 from './model22ans.ply'
import ANS23 from './model23ans.ply'
import ANS24 from './model24ans.ply'
import ANS31 from './obj1_recon_model3.ply'
// ./src/components/objects/Flower/reconstruct.ply

var mesh = null;
var geo;
var parentState;
var parentGlobal;
var res;
var selectedPC = false;
var selectedNoise = false;
var selectedInflate = false;
var selectedTwist = false;
var selectedSmooth = false;
var selectedSharpen = false;
var temp;

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
            modelType: 'Level1',
            twirl: 0,

            noiseApply: this.noise.bind(this),
            factorNoise: 1.0/8.0,

            inflateApply: this.inflate.bind(this),
            factorInflate: 1.0/8.0,

            twistApply: this.twist.bind(this),
            factorTwist: 1.0,
            
            smoothApply: this.smooth.bind(this),
            iter: 1,
            delta: 1.00,

            sharpenApply: this.sharpen.bind(this),
            sharpenIter: 1,
            sharpenDelta: 1.00,

            uploadMesh: this.uploadMesh.bind(this),
            // selectedFile: null

            Object1: 'Type1-1',
            Object2: 'Type2-1',
            Object3: 'Type3-1',
            GetModel1: this.gallery1.bind(this),
            GetModel2: this.gallery2.bind(this),
            GetModel3: this.gallery3.bind(this),
            
            reset: this.reset.bind(this)
        };

        
        // Load object
        const loader = new PLYLoader();

        this.name = 'baseModel';
        const path = './src/components/objects/models/' + 'reconstruct.ply';

        async function fetchMesh() {
            const response = await fetch('https://final-3d-reconstruction.herokuapp.com/post/', {
            //const response = await fetch('http://127.0.0.1:5000/retrievemeshes/', {
            method: 'POST',
            headers: {
              //'Accept': 'application/json',
              'Content-Type': 'application/json'
            },
            //body: JSON.stringify({'meshID': 1})
            body: JSON.stringify([])
          });
          const content = await response.blob();
          return content;
        }
        fetchMesh().then(content => {
            //console.log(content); // fetched movies
            //MODEL,
            //var url = URL.createObjectURL(content);
             // https://sbcode.net/threejs/loaders-ply/
            loader.load(
                MODEL,
                function (geometry) {
                    geometry.computeVertexNormals()
                    // https://stackoverflow.com/questions/25735128/three-js-show-single-vertex-as-ie-a-dot
                    // https://dev.to/maniflames/pointcloud-effect-in-three-js-3eic
                    
                    //const material = new MeshBasicMaterial({color: 0x5DADE2 });
                    const material = new MeshStandardMaterial();
                    mesh = new Mesh(geometry, material)
                    mesh.rotateX(-Math.PI / 2)
            
                    // ensuring mesh is inside unit cube or encompasses most of it
                    geometry.computeBoundingBox();
                    //console.log(geometry.boundingBox.max);
                    //console.log(geometry.boundingBox.min);
                    // console.log('before bounding box', geometry.boundingBox)
                    var max = 0.0;


                    max = Math.max( Math.abs(geometry.boundingBox.max.x), Math.abs(geometry.boundingBox.max.y),
                                    Math.abs(geometry.boundingBox.max.z), Math.abs(geometry.boundingBox.min.x),
                                    Math.abs(geometry.boundingBox.min.y), Math.abs(geometry.boundingBox.min.z),
                        )


                    // if (Math.abs(geometry.boundingBox.max.x) > max) max = geometry.boundingBox.max.x
                    // if (Math.abs(geometry.boundingBox.max.y) > max) max = geometry.boundingBox.max.y
                    // if (Math.abs(geometry.boundingBox.max.z) > max) max = geometry.boundingBox.max.z
                    // if (Math.abs(geometry.boundingBox.min.x) > max) max = geometry.boundingBox.min.x
                    // if (Math.abs(geometry.boundingBox.min.y) > max) max = geometry.boundingBox.min.y
                    // if (Math.abs(geometry.boundingBox.min.z) > max) max = geometry.boundingBox.min.z
                    //max = Math.abs(max);
                    //console.log("max: ",max)
                    var scale = 1.0/max;
                    // geometry.scale(scale, scale, scale);
                    // geometry.computeBoundingBox();
                    // console.log(geometry);
            
                    geo = geometry;
                    parent.add(mesh)
            
                },
                // function(object){
                //     parent.add( object );
                // },
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

        // 64, 96, or 128
        folders['PointCloudGeneration'].add(this.state, 'resolution', [64, 96, 128]);
        folders['PointCloudGeneration'].add(this.state, 'modelType', ['Level1', 'Level2', 'Level3']);
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

        var fold = filter.addFolder('smooth')
        fold.add(this.state, 'iter', 0, 3000)
        fold.add(this.state, 'delta', 0.00, 30)
        fold.add(this.state, 'smoothApply');

        var fold = filter.addFolder('sharpen')
        fold.add(this.state, 'sharpenIter', 0, 3000)
        fold.add(this.state, 'sharpenDelta', 0.00, 30.00)
        fold.add(this.state, 'sharpenApply');

        this.state.gui.add(this.state, 'uploadMesh');

        var fold = this.state.gui.addFolder('Gallery')

        var subFold = fold.addFolder('Model Type 1')
        subFold.add(this.state, 'Object1', ['Type1-1'])
        subFold.add(this.state, 'GetModel1')

        var subFold = fold.addFolder('Model Type 2')
        subFold.add(this.state, 'Object2', ['Type2-1', 'Type2-2','Type2-3', 'Type2-4'])
        subFold.add(this.state, 'GetModel2')

        var subFold = fold.addFolder('Model Type 3')
        subFold.add(this.state, 'Object3', ['Type3-1', 'Type3-2','Type3-3']);
        subFold.add(this.state, 'GetModel3')

        this.state.gui.add(this.state, 'reset');

    }

    // randomly select random set of vertices
    async pointCloudRand() {
        // send 30,000 vertices
        // send integer of number of vertices the user selected

        // will need textbox for reconstruction accuracy

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
        const samples = Math.round(parentState.numSamples);
        var pointCloud = [];
        var tracker = []
        const resolution = Math.round(this.state.resolution);
        const type = this.state.modelType;
       // console.log(samples)

        for (var i = 0; i < samples; i++){
            var rand = Math.floor(Math.random() * numVertices);
            while(tracker.includes(rand)) rand = Math.floor(Math.random() * numVertices);
            pointCloud.push([pos.array[rand * 3], pos.array[(rand * 3) + 1], pos.array[(rand * 3) + 2]]);
            tracker.push(rand);
        }
        //console.log(pointCloud);
        Swal.fire({
            title: 'Loading the newly created mesh'
        });
        Swal.showLoading();
       

        function delay(delayInms) {
            return new Promise(resolve => {
                setTimeout(() => {
                resolve(2);
                }, delayInms);
            });
        }
        // await delay(300);

        async function fetchMesh() {
        const response = await fetch('https://final-3d-reconstruction.herokuapp.com/post/', {
        //const response = await fetch('http://127.0.0.1:5000/post/', {
        method: 'POST',
        headers: {
            //'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({'data': JSON.stringify(pointCloud),
                                'resolution': resolution,
                            'type': type})
        //body: JSON.stringify({'data': pointCloud}),
        });
        const content = await response.blob();
        return content;
    }
    
    await fetchMesh().then(content => {
        //console.log(content); // fetched movies
        var url = URL.createObjectURL(content);
        res = new Result(parentGlobal, url)
    });


    Swal.close()
    parentGlobal.reconstructionComputed();

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

    smooth(){
        const iter = this.state.iter;
        const delta = -1.0 * this.state.delta;
        
        const n_vertices = geo.attributes.position.count;
        var geoProper = new Geometry().fromBufferGeometry( geo );

        //console.log(delta);
        var vertsCopy = [];
        for(var i = 0; i < n_vertices; i++){
            vertsCopy.push(this.getVertex(i));
        }
        for(var i = 0; i < iter; i++){
            for (var j = 0; j < n_vertices; j++){
                var v = this.getVertex(j);
                var neighbours = this.neighbouringVerts(j, geoProper);
                var weights = new Vector3(0, 0, 0);
               
                var num_neighbours = neighbours.size;
                for(var k = 0; k < num_neighbours; k++){
                    weights.add(this.getVertex(neighbours[k]));
                }

                let temp = new Vector3(-1*(v).getComponent(0)* num_neighbours, -1*(v).getComponent(1)* num_neighbours, -1*(v).getComponent(2)* num_neighbours);
                weights.add(temp);
                weights.multiplyScalar(delta);

                vertsCopy[j].add(weights);
            }

            //update original mesh
            for(let j = 0; j < n_vertices; j++){
                this.updateVertex(j, vertsCopy[j]);
            }
        }
        // console.log('Smooth complete')

        // geo.computeBoundingBox();
        // console.log('before bounding box', geo.boundingBox);
        this.boundMesh();
    }

    sharpen(){
        const iter = this.state.iter;
        const delta = this.state.delta;
        const n_vertices = geo.attributes.position.count;
        var geoProper = new Geometry().fromBufferGeometry( geo );

        var vertsCopy = [];
        for(var i = 0; i < n_vertices; i++){
            vertsCopy.push(this.getVertex(i));
        }
        for(var i = 0; i < iter; i++){
            for (var j = 0; j < n_vertices; j++){
                var v = this.getVertex(j);
                var neighbours = this.neighbouringVerts(j, geoProper);
                var weights = new Vector3(0, 0, 0);
               
                var num_neighbours = neighbours.size;
                for(var k = 0; k < num_neighbours; k++){
                    weights.add(this.getVertex(neighbours[k]));
                }

                let temp = new Vector3(1*(v).getComponent(0)* num_neighbours, 1*(v).getComponent(1)* num_neighbours, 1*(v).getComponent(2)* num_neighbours);
                weights.add(temp);
                weights.multiplyScalar(delta);

                vertsCopy[j].add(weights);
            }

            //update original mesh
            for(let j = 0; j < n_vertices; j++){
                this.updateVertex(j, vertsCopy[j]);
            }
        }
        console.log('Sharpen complete')

        // geo.computeBoundingBox();
        // console.log('before bounding box', geo.boundingBox);
        this.boundMesh();
    }

    async uploadMesh(){
        //swal("Hello world!");
    // https://sweetalert2.github.io/#download
    const { value: file } = await Swal.fire({
        title: 'Select a .ply file to upload',
        input: 'file',
        inputAttributes: {
          'aria-label': 'Upload your mesh, only use .ply files',
          'accept': '.ply'
        }
      })
      
      if (file) {
        const reader = new FileReader()

        const loader = new PLYLoader();
        var url = URL.createObjectURL(file);
        loader.load(
            url,
            function (geometry) {
                // console.log(geometry)
                geometry.computeVertexNormals()
                // https://stackoverflow.com/questions/25735128/three-js-show-single-vertex-as-ie-a-dot
                // https://dev.to/maniflames/pointcloud-effect-in-three-js-3eic
                const material = new MeshStandardMaterial({color: 0x5DADE2 });
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
                
                
                const object = parentGlobal.getObjectByProperty( 'uuid', parentGlobal.children[2]['uuid']);
                // referencing https://discourse.threejs.org/t/correctly-remove-mesh-from-scene-and-dispose-material-and-geometry/5448/2
                object.geometry.dispose();
                object.material.dispose();
                parentGlobal.remove( object );

                parentGlobal.add(mesh)
                // console.log(parentGlobal)
        
            },
            (xhr) => {
                console.log((xhr.loaded / xhr.total) * 100 + '% loaded')
            },
            (error) => {
                console.log(error)
            }
        )

      }
      //console.log(file)
    }


    gallery1(){
        //console.log('Gallery1')
        var type = this.state.Object1;

        var name = TYPE11;
        this.deleteMeshes()
        this.displayMesh(name, false)
        this.displayMesh(ANS11, true)
    }

    gallery2(){
        //console.log('Gallery2')
        var type = this.state.Object2;

        var name;
        var name2;
        if(type == 'Type2-1'){
            name = TYPE21;
            name2 = ANS21;
        }else if(type == 'Type2-2'){
            name = TYPE22;
            name2 = ANS22;
        }else if(type == 'Type2-3'){
            name = TYPE23;
            name2 = ANS23;
        }else if(type == 'Type2-4'){
            name = TYPE24;
            name2 = ANS24;
        }
        this.deleteMeshes()
        this.displayMesh(name, false)
        this.displayMesh(name2, true);
    }

    gallery3(){
        //console.log('Gallery3')
        var type = this.state.Object3;

        var name;
        if(type == 'Type3-1'){
            name = TYPE31;
        }else if(type == 'Type3-2'){
            name = TYPE32;
        }else if(type == 'Type3-3'){
            name = TYPE33;
        }
        this.deleteMeshes()
        this.displayMesh(name, false)
        this.displayMesh(ANS31, true)
    }

    reset(){
        // console.log(parentGlobal)
        this.deleteMeshes()
        this.displayMesh(MODEL)
        // console.log(parentGlobal)
    }

    deleteMeshes(){
        if(parentGlobal.children.length == 4){
            // console.log(parentGlobal.children[3]['uuid'])
            var object = parentGlobal.getObjectByProperty( 'uuid', parentGlobal.children[3]['uuid']);
            // referencing https://discourse.threejs.org/t/correctly-remove-mesh-from-scene-and-dispose-material-and-geometry/5448/2
            object.geometry.dispose();
            object.material.dispose();
            parentGlobal.remove( object );

            object = parentGlobal.getObjectByProperty( 'uuid', parentGlobal.children[2]['uuid']);
            // referencing https://discourse.threejs.org/t/correctly-remove-mesh-from-scene-and-dispose-material-and-geometry/5448/2
            object.geometry.dispose();
            object.material.dispose();
            parentGlobal.remove( object );
            
            // console.log(parentGlobal.children);
        }else{
            object = parentGlobal.getObjectByProperty( 'uuid', parentGlobal.children[2]['uuid']);
            // referencing https://discourse.threejs.org/t/correctly-remove-mesh-from-scene-and-dispose-material-and-geometry/5448/2
            object.geometry.dispose();
            object.material.dispose();
            parentGlobal.remove( object );
        }
    }

    displayMesh(name, offset){
        const loader = new PLYLoader();
        loader.load(
            name,
            function (geometry) {
                geometry.computeVertexNormals()
                // https://stackoverflow.com/questions/25735128/three-js-show-single-vertex-as-ie-a-dot
                // https://dev.to/maniflames/pointcloud-effect-in-three-js-3eic
                
                //const material = new MeshBasicMaterial({color: 0x5DADE2 });
                var material;
                if(offset) material = new MeshStandardMaterial({color: 0x5DADE2 });
                else  material = new MeshStandardMaterial();
                mesh = new Mesh(geometry, material)
                mesh.rotateX(-Math.PI / 2)
                if (offset) mesh.translateX(3)
        
                // ensuring mesh is inside unit cube or encompasses most of it
                geometry.computeBoundingBox();
                //console.log(geometry.boundingBox.max);
                //console.log(geometry.boundingBox.min);
                // console.log('before bounding box', geometry.boundingBox)
                var max = 0.0;


                max = Math.max( Math.abs(geometry.boundingBox.max.x), Math.abs(geometry.boundingBox.max.y),
                                Math.abs(geometry.boundingBox.max.z), Math.abs(geometry.boundingBox.min.x),
                                Math.abs(geometry.boundingBox.min.y), Math.abs(geometry.boundingBox.min.z),
                    )

                var scale = 1.0/max;
                geometry.scale(scale, scale, scale);
                geometry.computeBoundingBox();
                // console.log(geometry);
        
                geo = geometry;
                parentGlobal.add(mesh)
        
            },
            
            (xhr) => {
                console.log((xhr.loaded / xhr.total) * 100 + '% loaded')
            },
            (error) => {
                console.log(error)
            }
        )
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
    // get vertex of mesh at index i returns Vector3
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
    boundMesh(){
        geo.computeBoundingBox();
        // console.log('before bounding box', geo.boundingBox)
        var max = 0.0;
        max = Math.max( Math.abs(geo.boundingBox.max.x), Math.abs(geo.boundingBox.max.y),
                                    Math.abs(geo.boundingBox.max.z), Math.abs(geo.boundingBox.min.x),
                                    Math.abs(geo.boundingBox.min.y), Math.abs(geo.boundingBox.min.z),
                        )
        var scale = 1.0/max;
        geo.scale(scale, scale, scale);
        geo.computeBoundingBox();
        // console.log('after bounding box', geo.boundingBox)

        
    }
}



export default Flower;
