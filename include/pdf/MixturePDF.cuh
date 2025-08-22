#ifndef RENDERERPARALLEL_MIXTUREPDF_CUH
#define RENDERERPARALLEL_MIXTUREPDF_CUH

#include <pdf/CosinePDF.cuh>
#include <pdf/HittablePDF.cuh>

namespace renderer {
    class MixturePDF {
    private:
        typedef struct PDFTypeInfo {
            PDFType type;
            size_t index;
        } PDFTypeInfo;

        //PDF列表，前半部分为余弦PDF，后半部分为物体PDF
        PDFTypeInfo infoArray[32] {};
        size_t pdfCount;

        //将需要混合的PDF数组作为成员指针
        const CosinePDF * cosinePDFs;
        const HittablePDF * hittablePDFs;

    public:
        __device__ MixturePDF(const CosinePDF * cosinePDFs, const HittablePDF * hittablePDFs,
               Uint32 cosinePDFCount, Uint32 hittablePDFCount) :
               cosinePDFs(cosinePDFs), hittablePDFs(hittablePDFs), pdfCount(0) {
            for (size_t i = 0; i < hittablePDFCount; i++) {
                infoArray[pdfCount].type = PDFType::HITTABLE;
                infoArray[pdfCount++].index = i;
            }
            for (size_t i = 0; i < cosinePDFCount; i++) {
                infoArray[pdfCount].type = PDFType::COSINE;
                infoArray[pdfCount++].index = i;
            }
        }

        //当前支持球体和平行四边形作为采样物体
        __device__ Vec3 generate(curandState * state, const Sphere ** spheres, const Parallelogram ** parallelograms) const {
            //从PDF列表中随机选择一个
            const int randomIndex = randomIntDevice(state, 0, static_cast<int>(pdfCount) - 1);
            switch (infoArray[randomIndex].type) {
                case PDFType::COSINE:
                    return cosinePDFs[infoArray[randomIndex].index].generate(state);
                case PDFType::HITTABLE:
                    return hittablePDFs[infoArray[randomIndex].index].generate(state, spheres, parallelograms);
                default:
                    return Vec3();
            }
        }

        __device__ double value(const Sphere ** spheres, const Parallelogram ** parallelograms, const Vec3 &vec) const {
            //求所有PDF的加权平均值
            const double weight = 1.0 / static_cast<int>(pdfCount);

            double sum = 0.0;
            for (size_t i = 0; i < pdfCount; i++) {
                switch (infoArray[i].type) {
                    case PDFType::COSINE:
                        sum += weight * cosinePDFs[infoArray[i].index].value(vec);
                        break;
                    case PDFType::HITTABLE:
                        sum += weight * hittablePDFs[infoArray[i].index].value(spheres, parallelograms, vec);
                        break;
                    default:
                        return 0.0;
                }
            }
            return sum;
        }
    };
}

#endif //RENDERERPARALLEL_MIXTUREPDF_CUH
