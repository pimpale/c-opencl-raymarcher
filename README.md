# C OpenCL Ray Marcher

### Installation Instructions
X11 and OpenCL are required to be installed, and can be found easily in most linux repositories. To build, simply clone the repository and run `build.sh` in the directory. Clang or GCC will do. `clrenderer` will be generated. 

### Run Instructions

```./clrenderer <shadername>```
Two shader files are provided: sierpinski and warp. Sierpinski renders a 3d sierpinski triangle, while warp renders a black hole. These shaders can be taxing on the GPU, so if you run into trouble, recompile with a larger SCALE in main.c. This will decrease resolution.
The camera can be moved with the arrow keys, and rotated with the w,a,s,d,e, and q keys.

### Screenshots
#### Sierpinski
![sierpinski](./res/screenshots/sierpinski.png?raw=true "animated")
#### Black hole
![warp](./res/screenshots/warp.gif?raw=true "animated")
