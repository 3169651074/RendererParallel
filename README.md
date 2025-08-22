# RendererParallel
A simple ray tracer implemented by pure CUDA

## Build
此仓库中包含的源代码可直接在Windows和Linux上编译  
如果在Windows上编译，请确保依赖库（SDL2）所需文件已经被正确放置：bin目录中包含.dll动态库文件，lib目录中包含静态库文件。  
如果在Linux上编译，请先通过系统包管理器安装SDL2  

### Windows
克隆仓库后使用CMake编译，注意必须使用Visual Studio工具链，确保Visual Studio和C++桌面开发工具包、win10/11 SDK已安装

### Ubuntu/Debian
```
sudo apt update && sudo apt install libsdl2-dev libsdl2-image-dev
```