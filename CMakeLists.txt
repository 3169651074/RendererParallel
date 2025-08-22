#在Windows上使用MSVC编译
#在Linux上需要添加编译参数-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
cmake_minimum_required(VERSION 4.0)
set(EXECUTABLE_NAME "RendererParallel")
project(${EXECUTABLE_NAME} LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_ARCHITECTURES 89)

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin")
#set(LIBRARY_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/lib")
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})

include_directories("${CMAKE_SOURCE_DIR}/include")

if (WIN32)
    link_directories("${CMAKE_SOURCE_DIR}/lib")
else ()
    find_package(SDL2 REQUIRED)
    find_package(SDL2_image REQUIRED)
endif ()

#file(GLOB_RECURSE SRCS "${CMAKE_SOURCE_DIR}/src/*.cu")
add_executable(${EXECUTABLE_NAME} #${SRCS}
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

if (WIN32)
    #MSVC编译器必须指定以UTF-8编码读取文件，否则在Windows上需要将所有文件的编码设置为UTF-8 with BOM
    target_compile_options(${EXECUTABLE_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    #必须启用多文件链接支持
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()

#链接SDL2库
if (WIN32)
    target_link_libraries(${EXECUTABLE_NAME} PUBLIC SDL2main)
endif ()
target_link_libraries(${EXECUTABLE_NAME} PUBLIC SDL2 SDL2_image)
