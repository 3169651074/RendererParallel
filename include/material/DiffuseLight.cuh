#ifndef RENDERERPARALLEL_DIFFUSELIGHT_CUH
#define RENDERERPARALLEL_DIFFUSELIGHT_CUH

#include <basic/Ray.cuh>
#include <basic/Color3.cuh>

namespace renderer {
    class DiffuseLight {
    private:
        Color3 light;

    public:
        __host__ explicit DiffuseLight(const Color3 & lightColor) : light(lightColor) {}

        __device__ Color3 emitted(const Ray & ray, const HitRecord & record) const {
//            if (record.hitFrontFace) {
//                return light;
//            } else {
//                return Color3();
//            }
            return light;
        }
    };
}

#endif //RENDERERPARALLEL_DIFFUSELIGHT_CUH
