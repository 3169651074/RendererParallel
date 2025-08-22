#ifndef RENDERERPARALLEL_ORTHONORMALBASE_CUH
#define RENDERERPARALLEL_ORTHONORMALBASE_CUH

#include <basic/Vec3.cuh>

namespace renderer {
    //正交基工具类，设备端和主机端通用
    class OrthonormalBase {
    public:
        Vec3 elements[3];

        //根据向量构造正交基，不要求传入单位向量，可以指定传入的向量作为局部坐标系的轴的下标
        __host__ __device__ OrthonormalBase(const Vec3 & vec, int axis) {
//            if (axis < 0 || axis > 2) {
//                throw std::runtime_error("Axis out of bound!");
//            }

            //将vec的单位向量作为局部坐标系的z轴（axis = 2）
            elements[axis] = vec.unitVector();

            //使用坐标轴构造不平行于vec的“任意”向量a
            Vec3 a;
            if (elements[axis][0] > 0.9) {
                //传入向量和x轴平行，则将y轴作为a
                a[1] = 1.0;
            } else {
                //将x轴作为a
                a[0] = 1.0;
            }

            if (axis == 0 || axis == 2) {
                elements[1] = Vec3::cross(elements[axis], a).unitVector();
                elements[2 - axis] = Vec3::cross(elements[axis], elements[1]);
            } else { //axis == 1
                elements[2] = Vec3::cross(elements[axis], a).unitVector();
                elements[0] = Vec3::cross(elements[axis], elements[1]);
            }
        }

        //直接使用传入的正交向量（允许非单位向量）构造正交基
        //cameraU --> x; v --> y; w --> z
        __host__ __device__ explicit OrthonormalBase(const Vec3 & x = Vec3(1.0, 0.0, 0.0), const Vec3 & y = Vec3(0.0, 1.0, 0.0), const Vec3 & z = Vec3(0.0, 0.0, 1.0)) {
            elements[0] = x.unitVector();
            elements[1] = y.unitVector();
            elements[2] = z.unitVector();
        }

        //拷贝构造，赋值运算符和析构函数
        __host__ __device__ OrthonormalBase(const OrthonormalBase & obj) {
            for (size_t i = 0; i < 3; i++) {
                elements[i] = obj.elements[i];
            }
        }

        __host__ __device__ OrthonormalBase & operator=(const OrthonormalBase & obj) {
            if (this == &obj) return *this;
            for (size_t i = 0; i < 3; i++) {
                elements[i] = obj.elements[i];
            }
            return *this;
        }

        __host__ __device__ ~OrthonormalBase() {}

        //将局部空间中的向量origin变换到世界空间
        __host__ __device__ Vec3 transformToWorld(const Vec3 & origin) const {
            Vec3 ret;
            for (size_t i = 0; i < 3; i++) {
                //将origin的各轴向分量（标量值）和轴向量（单位方向向量）数乘
                ret += elements[i] * origin[i];
            }
            return ret;
        }

        //将世界空间中的向量 worldVec 变换到此正交基定义的局部空间
        __host__ __device__ Vec3 transformToLocal(const Vec3 & origin) const {
            //局部坐标的分量等于世界向量在各个基向量上的投影，这个投影可以通过点积来计算
            Vec3 ret;
            for (size_t i = 0; i < 3; i++) {
                ret[i] = Vec3::dot(elements[i], origin);
            }
            return ret;
        }
    };
}

#endif //RENDERERPARALLEL_ORTHONORMALBASE_CUH
