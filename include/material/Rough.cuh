#ifndef RENDERERPARALLEL_ROUGH_CUH
#define RENDERERPARALLEL_ROUGH_CUH

#include <basic/Ray.cuh>
#include <basic/Color3.cuh>

namespace renderer {
    /*
     * 粗糙材质类
     */
    class Rough {
    private:
        Color3 albedo;

    public:
        __host__ explicit Rough(const Color3 & albedo) : albedo(albedo) {}

        //粗糙材质：漫反射
        __device__ bool scatter(curandState * state, const Ray & in, const HitRecord & record, Color3 & attenuation, Ray & out) const {
            //随机选择一个反射方向
            Vec3 reflectDirection = (record.normalVector + Vec3::randomSpaceVectorDevice(state, 1.0)).unitVector();

            //随机的反射方向可能和法向量相互抵消，此时取消随机反射
            if (floatValueEquals(reflectDirection.lengthSquare(), Vec3::VECTOR_LENGTH_SQUARE_ZERO_EPSILON)) {
                reflectDirection = record.normalVector;
            }

            //从反射点出发构造反射光线
            out = Ray(record.hitPoint, reflectDirection, in.time);
            attenuation = albedo;
            return true;
        }

        /*
         * BRDF是一个函数，它能根据入射光线的方向，摄像机的方向和法线方向计算出特定方向上的反射光强度
         * Bidirectional Reflectance Distribution Function
         */
        __device__ Color3 evalBRDF(const Ray & in, const HitRecord & record) const {
            return albedo / PI;
        }

        //计算渲染方程中的余弦项
        __device__ double cosTheta(const Ray & out, const HitRecord & record) const {
            return max(0.0, Vec3::dot(record.normalVector, out.direction.unitVector()));
        }
    };
}

#endif //RENDERERPARALLEL_ROUGH_CUH
