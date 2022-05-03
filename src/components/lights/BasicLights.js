import { Group, SpotLight, AmbientLight, HemisphereLight, PointLight } from 'three';

class BasicLights extends Group {
    constructor(...args) {
        // Invoke parent Group() constructor with our args
        super(...args);
        const hemi = new HemisphereLight(0xffffbb, 0x080820, 1.1);
        this.add( hemi);

    }
}

export default BasicLights;
