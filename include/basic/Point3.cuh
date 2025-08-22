#ifndef RENDERERPARALLEL_POINT3_CUH
#define RENDERERPARALLEL_POINT3_CUH

#include <basic/Vec3.cuh>

namespace renderer {
    /*
     * 空间点类，主机端和设备端通用
     */
    class Point3 {
    private:
        double elements[3] {};

    public:
        __host__ __device__ explicit Point3(double x = 0.0, double y = 0.0, double z = 0.0) {
            elements[0] = x; elements[1] = y; elements[2] = z;
        }

        //向量转点
        __host__ __device__ explicit Point3(const Vec3 & obj) {
            for (size_t i = 0; i < 3; i++) { elements[i] = obj[i]; }
        }

        __host__ __device__ Point3(const Point3 & obj) {
            for (size_t i = 0; i < 3; i++) { elements[i] = obj[i]; }
        }

        __host__ __device__ Point3 & operator=(const Point3 & obj) {
            if (this == &obj) return *this;
            for (size_t i = 0; i < 3; i++) { elements[i] = obj[i]; }
            return *this;
        }

        __host__ __device__ ~Point3() {}

        //成员访问函数

        __host__ __device__ double & operator[](size_t index) {
            return elements[index];
        }
        __host__ __device__ double operator[](size_t index) const {
            return elements[index];
        }

        // ====== 对象操作函数 ======

        __host__ __device__ double distanceSquare(const Point3 & anotherPoint) const {
            double sum = 0.0;
            for (size_t i = 0; i < 3; i++) {
                sum += (elements[i] - anotherPoint.elements[i]) * (elements[i] - anotherPoint.elements[i]);
            }
            return sum;
        }

        //设备函数需要调用cuda库中的数学函数，不加std前缀，编译器自动寻找对应版本的函数实现
        __host__ __device__ double distance(const Point3 & anotherPoint) const {
            return sqrt(distanceSquare(anotherPoint));
        }

        //点偏移
        __host__ __device__ Point3 & operator+=(const Vec3 & offset) {
            for (size_t i = 0; i < 3; i++) {
                elements[i] += offset[i];
            }
            return *this;
        }

        __host__ __device__ Point3 operator+(const Vec3 & offset) const {
            Point3 ret(*this); ret += offset; return ret;
        }

        __host__ __device__ Point3 & operator-=(const Vec3 & offset) {
            for (size_t i = 0; i < 3; i++) {
                elements[i] -= offset[i];
            }
            return *this;
        }

        __host__ __device__ Point3 operator-(const Vec3 & offset) const {
            Point3 ret(*this); ret -= offset; return ret;
        }

        //点转向量（向量转点通过构造方法）
        __host__ __device__ Vec3 toVector() const  {
            return Vec3(elements[0], elements[1], elements[2]);
        }

        // ====== 静态操作函数 ======

        __host__ __device__ static inline double distanceSquare(const Point3 & p1, const Point3 & p2) {
            return p1.distanceSquare(p2);
        }

        __host__ __device__ static inline double distance(const Point3 & p1, const Point3 & p2) {
            return sqrt(p1.distanceSquare(p2));
        }

        __host__ __device__ static inline Vec3 constructVector(const Point3 & from, const Point3 & to) {
            Vec3 ret;
            for (size_t i = 0; i < 3; i++) {
                ret[i] = to.elements[i] - from.elements[i];
            }
            return ret;
        }

        // ====== 类封装函数 ======

        __host__ std::string toString() const {
            char buffer[TOSTRING_BUFFER_SIZE] = { 0 };
            snprintf(buffer, TOSTRING_BUFFER_SIZE, "Point3: (%.4lf, %.4lf, %.4lf)", elements[0], elements[1], elements[2]);
            return {buffer};
        }
    };
}

#endif //RENDERERPARALLEL_POINT3_CUH
