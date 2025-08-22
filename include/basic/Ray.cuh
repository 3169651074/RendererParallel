#ifndef RENDERERPARALLEL_RAY_CUH
#define RENDERERPARALLEL_RAY_CUH

#include <basic/Point3.cuh>
#include <Structs.cuh>

namespace renderer {
    /*
     * 光线类。P(t) = A + tB
     * 主机端和设备端通用
     */
    class Ray {
    public:
        Point3 origin;
        Vec3 direction;
        double time;

        __host__ __device__ explicit Ray(const Point3 & origin = Point3(), const Vec3 & direction = Vec3(1.0, 0.0, 0.0),
                     double time = 0.0) : origin(origin), direction(direction), time(time) {}

        __host__ __device__ Ray(const Ray & obj) {
            origin = obj.origin;
            direction = obj.direction;
            time = obj.time;
        }

        __host__ __device__ Ray & operator=(const Ray & obj) {
            if (this == &obj) return *this;
            origin = obj.origin;
            direction = obj.direction;
            time = obj.time;
            return *this;
        }

        __host__ __device__ ~Ray() {}

        // ====== 对象操作函数 ======

        __host__ __device__ Point3 at(double t) const {
            return origin + t * direction;
        }

        // ====== 类封装函数 ======

        __host__ std::string toString() const {
            char buffer[TOSTRING_BUFFER_SIZE] = { 0 };
            snprintf(buffer, TOSTRING_BUFFER_SIZE, "Ray: Origin = (%.4lf, %.4lf, %.4lf), Direction = (%.4lf, %.4lf, %.4lf)",
                     origin[0], origin[1], origin[2], direction[0], direction[1], direction[2]);
            return {buffer};
        }
    };
}

#endif //RENDERERPARALLEL_RAY_CUH
