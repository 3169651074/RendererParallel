#ifndef RENDERERPARALLEL_DIELECTRIC_CUH
#define RENDERERPARALLEL_DIELECTRIC_CUH

#include <basic/Ray.cuh>
#include <basic/Color3.cuh>

namespace renderer {
    class Dielectric {
    private:
        Color3 albedo;
        double refractiveIndex;

        //使用Schlick近似计算反射率
        __device__ static double reflectance(double cosine, double refractiveIndex) {
            double r0 = (1.0 - refractiveIndex) / (1.0 + refractiveIndex);
            r0 = r0 * r0;
            return r0 + (1.0 - r0) * std::pow((1.0 - cosine), 5.0);
        }

        //计算折射光线，要求i和n都是单位向量，需要根据光线的入射方向决定相对折射率
        __device__ Vec3 refract(curandState * state, const Vec3 & i, const Vec3 & n, bool isFrontFace) const {
            const double cosTheta = Vec3::dot(-i, n);
            const double sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);
            const double rate = isFrontFace ? 1.0 / refractiveIndex : refractiveIndex * 1.0; //根据入射方向确定折射率

            //确定是否发生全反射
            if (sinTheta * rate > 1.0 || reflectance(cosTheta, refractiveIndex) > randomDoubleDevice(state)) {
                //全反射
                return i - 2 * Vec3::dot(i, n) * n;
            } else {
                //折射
                const Vec3 perpendicular = rate * (i + cosTheta * n); //和法向量垂直的折射分量
                const Vec3 parallel = -std::sqrt(std::abs(1.0 - perpendicular.lengthSquare())) * n;
                return perpendicular + parallel;
            }
        }

    public:
        __host__ explicit Dielectric(double refractiveIndex = 1.0) ://材质不吸收光
                albedo(Color3(1.0, 1.0, 1.0)), refractiveIndex(refractiveIndex) {}

        __device__ bool scatter(curandState * state, const Ray &in, const HitRecord &record, Color3 & attenuation, Ray & out) const {
            //单位化输入向量
            const Vec3 i = in.direction.unitVector();
            //计算折射向量
            const Vec3 r = refract(state, i, record.normalVector, record.hitFrontFace);
            //构造折射光线
            out = Ray(record.hitPoint, r.unitVector(), in.time);
            attenuation = albedo;
            return true;
        }
    };
}

#endif //RENDERERPARALLEL_DIELECTRIC_CUH
