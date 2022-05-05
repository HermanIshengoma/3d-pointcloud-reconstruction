import * as Dat from 'dat.gui';
import { Scene, Color, GridHelper } from 'three';
import { Flower, Land, Models, Result} from 'objects';
import { BasicLights } from 'lights';
// import { Result } from '../objects';

class SeedScene extends Scene {
    constructor(meshObj) {
        // Call parent Scene() constructor
        super();

        // Init state
        this.state = {
            gui: new Dat.GUI(), // Create GUI for scene
            rotationSpeed: 1,
            updateList: [],
            numSamples: 1000,
            reconstructed: false,
        };

        // Set background to a nice color
        // this.background = new Color(0x7ec0ee);
        // this.background = new Color(0x666666);
        this.background = new Color(0x0);
        // new Result()

        // Populate GUI
        var folders = {};
        // this.state.gui.add(this.state, 'rotationSpeed', -5, 5);
        const pcGen = this.state.gui.addFolder('PointCloudGeneration');
        folders["PointCloudGeneration"] = pcGen;
        pcGen.add(this.state, 'numSamples', 1000, 10000);


        // Add meshes to scene
        const model = new Flower(this, meshObj, this.state, folders);
        //const model = new Models();
        const lights = new BasicLights();
        
        
        this.add(model, lights);
    }

    addToUpdateList(object) {
        this.state.updateList.push(object);
    }

    reconstructionComputed(){
        this.state.reconstructed = true;
        
    }

    update(timeStamp) {
        const { rotationSpeed, updateList } = this.state;
        //this.rotation.y = (rotationSpeed * timeStamp) / 10000;

        // Call update for each object in the updateList
        for (const obj of updateList) {
            obj.update(timeStamp);
        }
    }
}

export default SeedScene;
