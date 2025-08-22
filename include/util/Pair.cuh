#ifndef RENDERERPARALLEL_PAIR_CUH
#define RENDERERPARALLEL_PAIR_CUH

#include <Global.cuh>

namespace renderer {
    /*
     * 用于替代std::pair的工具类，std::pair没有提供设备端的构造函数和赋值运算符
     * 常规情况下编译器默认生成的6个成员函数没有device版本，必须手动提供，包括构造，拷贝构造，赋值运算符和析构
     * 对于GPU上可用的POD类型，构造和赋值需要逐个成员赋值，析构函数置空
     */
    template <typename T1, typename T2>
    class Pair {
    public:
        T1 first;
        T2 second;

        //通用构造函数和赋值运算符
        __host__ __device__ explicit Pair(const T1 & first = T1(), const T2 & second = T2()) :
            first(first), second(second) {}

        __host__ __device__ Pair(const Pair & obj) {
            first = obj.first;
            second = obj.second;
        }

        __host__ __device__ Pair& operator=(const Pair & obj) {
            if (this == &obj) return *this;
            first = obj.first;
            second = obj.second;
            return *this;
        }

        __host__ __device__ ~Pair() {}
    };
}

#endif //RENDERERPARALLEL_PAIR_CUH
