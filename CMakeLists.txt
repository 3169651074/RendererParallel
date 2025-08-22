cmake_minimum_required(VERSION 4.0)
set(EXECUTABLE_NAME "RendererParallel")

set(CMAKE_CUDA_ARCHITECTURES 89)
project(${EXECUTABLE_NAME} LANGUAGES CUDA)

#必须加上cmake编译参数
#-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
set(CMAKE_CUDA_STANDARD 11)
#强制要求 CUDA 编译器必须支持指定的C++标准，如果不支持则编译报错，而不是自动降级到较低的标准
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin")
set(LIBRARY_OUTPUT_PATH ${EXECUTABLE_OUTPUT_PATH})

if (WIN32)
    link_directories("${CMAKE_SOURCE_DIR}/lib")
else ()
    #sudo apt install libspdlog-dev && sudo apt install libfmt-dev
    #find_package(spdlog REQUIRED)
    #find_package(fmt REQUIRED)
    find_package(SDL2 REQUIRED)
endif ()

include_directories("${CMAKE_SOURCE_DIR}/include")

add_executable(${EXECUTABLE_NAME}
        src/Main.cu
        include/Global.cuh
        include/Render.cuh
        include/basic/Vec3.cuh
        include/basic/Point3.cuh
        include/basic/Color3.cuh
        include/util/Range.cuh
        include/basic/Ray.cuh
        include/Structs.cuh
        include/hittable/Sphere.cuh
        include/box/BoundingBox.cuh
        include/box/BVHTree.cuh
        include/hittable/Triangle.cuh
        include/material/Metal.cuh
        include/material/Rough.cuh
        include/Camera.cuh
        src/Camera.cu
        src/Render.cu
        src/Kernel.cu
        include/util/Pair.cuh
        src/Global.cu
        include/Examples.hpp
        src/Examples.cu
        include/hittable/Parallelogram.cuh
        include/util/OrthonormalBase.cuh
        include/material/Dielectric.cuh
        include/material/DiffuseLight.cuh
        include/hittable/Box.cuh
        include/hittable/Transform.cuh
        include/util/Matrix.cuh
        src/util/Matrix.cu
        include/pdf/CosinePDF.cuh
        include/pdf/HittablePDF.cuh
        include/pdf/MixturePDF.cuh
)

#允许设备代码跨源文件链接，支持复杂设备端调用和结构
set_target_properties(${EXECUTABLE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#链接动态库
if (WIN32)
    target_link_libraries(${EXECUTABLE_NAME} PUBLIC mingw32 SDL2main)
endif ()
target_link_libraries(${EXECUTABLE_NAME} PUBLIC SDL2 SDL2_image)

#target_link_libraries(${EXECUTABLE_NAME} PUBLIC spdlog fmt)
