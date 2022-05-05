# COS 426 Final Project -- 3D Reconstruction of Point Clouds via Neural SDFs

[Online Demo](https://hermanishengoma.github.io/3d-pointcloud-reconstruction/)

## Demo (Gallery)
The gallery folder contains our code to running the online demo. We provide a gallery that contains some results we trained locally. 

## Demo (Interactive)
The local_demo folder contains our code to running the demo interactively, and allows user to reconstruct meshes on their personal computer. Due to package sizes we were unable to deploy on Heroku. 
Open two terminals and cd into the backend and frontend folder respectively. Run "npm start" in the frontend folder and "flask run" in the backend folder. Then navigate the localhost url displayed on the frontend terminal.
We recommend creating a conda environment for packages: torch, numpy, scipy, plyfile, trimesh, pytorch-lightning, flask. Follow terminal instructions to install any additional packages.
For questions on running the code, feel free to email Gene (gchou@princeton.edu)

## References
For our SDF code, we largely borrow from DeepSDF and Convolutional Occupancy Networks. 
