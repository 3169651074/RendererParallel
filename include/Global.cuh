#ifndef RENDERERPARALLEL_GLOBAL_CUH
#define RENDERERPARALLEL_GLOBAL_CUH

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>

#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <stack>
#include <memory>
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>
#include <limits>
#include <array>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#undef INFINITY
#undef NULL

/*
 * 通用工具类：设备端和主机端通用
 * 类中的默认成员函数：编译器只会生成主机版本。如果类需要在主机端和设备端使用，则需要手动给出+空方法体
 * 如果类中的默认成员函数只被主机调用，则可不手动写
 *
 * 添加新图元类型：
 * 1. Renderer类中添加指针成员，新增图元枚举
 * 2. 添加参数到commitSceneData方法
 * 3. 在commitSceneData方法的实现中添加BVH构建宏调用，分配显存和拷贝数据语句
 * 4. 在freeSceneData方法的实现中添加释放显存语句
 * 5. 添加参数到constructBVHTree方法，并添加宏调用
 * 6. 添加参数到BVHTree::hit方法，并扩展switch语句
 * 7. 添加参数到Transform::hit，并扩展switch语句
 *
 * 添加新材质类型：
 * 1. Renderer类中添加指针成员，新增材质枚举
 * 2. 添加参数到commitSceneData方法
 * 3. 在commitSceneData方法的实现中添加分配显存和拷贝数据语句
 * 4. 在freeSceneData方法的实现中添加释放显存语句
 * 5. 扩展rayColor函数的switch语句
 */
namespace renderer {
    // ====== 数值常量 ======

    constexpr double FLOAT_VALUE_ZERO_EPSILON = 1e-5;
    constexpr double INFINITY = std::numeric_limits<double>::infinity();
    constexpr double PI = M_PI;
    constexpr Uint32 TOSTRING_BUFFER_SIZE = 200;

    // ====== 数学工具函数 ======

    __host__ __device__ inline double degreeToRadian(const double degree) {
        return degree * PI / 180.0;
    }

    __host__ __device__ inline double radianToDegree(const double radian) {
        return radian * 180.0 / PI;
    }

    //判断浮点数是否接近于0
    __host__ __device__ inline bool floatValueNearZero(const double val) {
        return abs(val) < FLOAT_VALUE_ZERO_EPSILON;
    }

    //判断两个浮点数是否相等
    __host__ __device__ inline bool floatValueEquals(const double v1, const double v2) {
        return abs(v1 - v2) < FLOAT_VALUE_ZERO_EPSILON;
    }

    // ====== 主机端随机数生成函数 ======

    //生成一个[0, 1)之间的浮点随机数
    __host__ inline double randomDoubleHost() {
        static std::random_device rd; //需要初始化随机设备，让每次运行都能生成不同的随机数
        static std::uniform_real_distribution<> distribution(0.0, 1.0);
        static std::mt19937 generator(rd());
        return distribution(generator);
    }

    //生成一个[min, max)之间的浮点随机数
    __host__ inline double randomDoubleHost(double min, double max) {
        return min + (max - min) * randomDoubleHost();
    }

    //生成一个[min, max]之间的整数随机数
    __host__ inline int randomIntHost(int min, int max) {
        //直接使用randomDouble()，以共享同一个静态生成器
        return static_cast<int>(randomDoubleHost(min, max + 1));
    }

    // ====== 设备端随机数生成函数 ======

    //生成一个[0, 1)之间的随机浮点数
    __device__ inline double randomDoubleDevice(curandState * state) {
        return curand_uniform_double(state);
    }

    //生成一个[a, b)之间的随机浮点数
    __device__ inline double randomDoubleDevice(curandState * state, double min, double max) {
        return randomDoubleDevice(state) * (max - min) + min;
    }

    //生成一个[min, max]之间的整数随机数
    __device__ inline int randomIntDevice(curandState * state, int min, int max) {
        return static_cast<int>(randomDoubleDevice(state, min, max + 1));
    }

#define arrayLengthOnPos(name) sizeof(name) / sizeof(name[0])

    // ====== SDL库包装函数 ======

#define releaseSDLResource(function, taskName) function; SDL_Log("%s...Done", taskName)
#define expandSDLColor(color) color.r, color.g, color.b, color.a

    typedef enum class TaskOnError {
        PRINT_MESSAGE, IGNORE, EXIT_PROGRAM
    } TaskOnError;

    extern void (*releaseSDLResourcesFunc)();

    //设置资源释放函数
    void SDL_registerReleaseResources(void (*f)());

    //检查SDL函数返回值
    void SDL_CheckErrorInt(int retVal, const char *taskName, TaskOnError errorTask = TaskOnError::EXIT_PROGRAM);
    void SDL_CheckErrorPtr(void *retVal, const char *taskName, TaskOnError errorTask = TaskOnError::EXIT_PROGRAM);

    // ====== CUDA库包装函数 ======

    //C++11支持__func__宏获取当前函数
#define cudaCheckError(call) _handle_error0(call, __FILE__, __func__, __LINE__)

    //检查cuda库函数返回值
    static inline void _handle_error0(cudaError_t err, const char * file, const char * function, int line, TaskOnError task = TaskOnError::EXIT_PROGRAM) {
        if (err != cudaSuccess) {
            SDL_LogError(SDL_LOG_CATEGORY_ERROR, "CUDA Error: %s, at function %s: in file %s line %d", cudaGetErrorString(err), function, file, line);
            exit(EXIT_FAILURE);
        }
    }
}

#endif //RENDERERPARALLEL_GLOBAL_CUH
