#ifndef RENDERERPARALLEL_METAL_CUH
#define RENDERERPARALLEL_METAL_CUH

#include <basic/Ray.cuh>
#include <basic/Color3.cuh>

namespace renderer {
    /*
     * 金属材质类
     */
    class Metal {
    private:
        Color3 albedo;
        double fuzz;

    public:
        __host__ explicit Metal(const Color3 & albedo = Color3(1.0, 1.0, 1.0), double fuzz = 0.0) :
                albedo(albedo), fuzz(fuzz)
        {
            this->fuzz = Range(0.0, 1.0).clamp(fuzz);
        }

        //金属材质不吸收光线，完全反射光线
        __device__ bool scatter(curandState * state, const Ray & in, const HitRecord & record, Color3 & attenuation, Ray & out) const {
            //计算反射光线方向向量（单位向量）
            const Vec3 v = in.direction;
            const Vec3 n = record.normalVector;
            Vec3 reflectDirection = (v - 2 * Vec3::dot(v, n) * n).unitVector();

            //应用反射扰动：在距离物体表面1单位处随机选取单位向量和反射向量相加，形成随机扰动
            if (fuzz > 0.0) {
                reflectDirection += fuzz * Vec3::randomSpaceVectorDevice(state, 1.0);
            }

            //构建反射光线，光线的时间属性不随传播而改变
            out = Ray(record.hitPoint, reflectDirection, in.time);
            attenuation = albedo;

            //防止由于计算精度导致out方向向内。使用点积检查反射光线是否和物体外法线的同侧，仅当二者同侧时有效
            return Vec3::dot(out.direction, record.normalVector) > 0.0;
        }
    };
}

#endif //RENDERERPARALLEL_METAL_CUH
