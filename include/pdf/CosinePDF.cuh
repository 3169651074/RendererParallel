#ifndef RENDERERPARALLEL_COSINEPDF_CUH
#define RENDERERPARALLEL_COSINEPDF_CUH

#include <util/OrthonormalBase.cuh>

namespace renderer {
    /*
     * 余弦概率密度，离法向量近的方向概率密度高
     * 需要在渲染函数中由GPU线程创建
     */
    class CosinePDF {
    private:
        //由法向量构造局部坐标系
        OrthonormalBase base;

    public:
        __device__ explicit CosinePDF(const Vec3 & normal) : base(normal, 2) {}

        __device__ ~CosinePDF() {}

        __device__ Vec3 generate(curandState * state) const {
            return base.transformToWorld(Vec3::randomCosineVectorDevice(state, 2, true));
        }

        __device__ double value(const Vec3 &vec) const {
            //保证概率密度不为负
            return max(0.0, Vec3::dot(vec.unitVector(), base.elements[2]) / PI);
        }
    };
}

#endif //RENDERERPARALLEL_COSINEPDF_CUH
