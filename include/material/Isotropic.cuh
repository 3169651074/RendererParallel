#ifndef RENDERERPARALLEL_ISOTROPIC_CUH
#define RENDERERPARALLEL_ISOTROPIC_CUH

#include <basic/Ray.cuh>
#include <basic/Color3.cuh>

namespace renderer {
    class Isotropic {
    private:
        Color3 albedo;

        bool isWrapped;

    public:
        __host__ explicit Isotropic(const Color3 & albedo, bool isWrapped = false) : albedo(albedo), isWrapped(isWrapped) {}

        /*
         * 体积雾材质本身没有一个固定的BRDF，因为它不是一个表面材质，而是一种体积现象
         * BRDF函数是用来描述光线在一个表面上反射特性的，它定义了从一个特定方向入射的光，在反射后会向各个方向传播的能量分布。它只关注表面的行为
         *
         * 而体积雾是三维空间中的一种介质，光线在其中会发生吸收（Absorption）、散射（Scattering）和自发光（Emission）等多种复杂的相互作用。
         * 这些现象不是由BRDF函数能简单描述的，而是通过体积渲染方程（Volume Rendering Equation）来求解
         *
         * 体积雾中的光线行为：
         *   散射（Scattering）：
         *     向前散射（Forward Scattering）: 光线被雾中的微粒偏转，继续向前传播。这使得靠近光源的区域看起来更亮，光束效应（God Rays）就是这种效果的体现。
         *     向后散射（Backward Scattering）: 光线被散射回光源方向，这让光源附近的区域变得更亮。
         *     各向同性散射（Isotropic Scattering）: 光线被均匀地散射到所有方向。
         *     各向异性散射（Anisotropic Scattering）: 光线被非均匀地散射，通常有一个主导方向，用亨尼-格林斯坦相位函数（Henyey-Greenstein Phase Function）来描述。
         *     p(cosθ) = (1−g^2) / (1+g^2-2gcosθ)^1.5次方
         *       其中，g是各向异性因子（Anisotropy Factor），其值范围在 -1 到 1 之间：g>0 表示以向前散射为主,g=0 表示各向同性散射。
         *   吸收（Absorption）: 雾中的介质会吸收一部分光能，导致光线强度随着传播距离的增加而衰减。
         *   自发光（Emission）: 体积雾自身可以发光，比如爆炸后产生的火焰光效。
         *
         * 通常用一个或多个参数来定义上述这些行为：
         *   密度（Density）: 影响散射和吸收的程度。
         *   吸收系数（Absorption Coefficient）：决定光被吸收的速率。
         *   散射系数（Scattering Coefficient）：决定光被散射的速率。
         *   各向异性因子（Anisotropy Factor）：定义散射的方向性。
         *
         * 体积雾依赖体积渲染方程和相位函数来描述光在介质中的复杂运动
         */
        __device__ bool scatter(curandState * state, const Ray & in, const HitRecord & record, Color3 & attenuation, Ray & out) const {
            out = Ray(record.hitPoint, Vec3::randomSpaceVectorDevice(state, 1.0), in.time);
            attenuation = albedo;
            return true;
        }

        __device__ double scatterPDF() const {
            return 1.0 / (4.0 * PI);
        }
    };
}

#endif //RENDERERPARALLEL_ISOTROPIC_CUH
