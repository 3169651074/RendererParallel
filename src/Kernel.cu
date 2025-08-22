#include <Render.cuh>

namespace renderer {
    __device__ Color3 rayColor(const Renderer * dev_renderer, const Camera * dev_cam, const Ray * ray, curandState * state)
    {
        HitRecord record;
        Ray currentRay(*ray);
        Color3 result(1.0, 1.0, 1.0);

        for (size_t currentIterateDepth = 0; currentIterateDepth < dev_cam->rayTraceDepth; currentIterateDepth++) {
            //每次追踪，都遍历所有球体，判断是否相交
            if (BVHTree::hit(dev_renderer->dev_tree, dev_renderer->dev_treeIndexArray,
                             dev_renderer->dev_spheres,
                             dev_renderer->dev_triangles,
                             dev_renderer->dev_parallelograms,
                             dev_renderer->dev_boxes,
                             dev_renderer->dev_transforms,
                             currentRay, Range(0.001, INFINITY), record))
            {
                //发生碰撞，调用材质的散射函数，获取下一次迭代的光线
                Ray out;
                Color3 attenuation;

                //优先处理光源材质
                if (record.materialType == MaterialType::DIFFUSE_LIGHT) {
                    return result * dev_renderer->dev_diffuseLightMaterials[record.materialIndex].emitted(currentRay, record);
                }

                //根据材质类型调用对应的散射函数
                switch (record.materialType) {
                    case MaterialType::ROUGH: {
                        //如果没有设置直接采样物体，则不使用重要性采样
                        if (!dev_renderer->isDirectSample) {
                            dev_renderer->dev_roughMaterials[record.materialIndex].scatter(state, currentRay, record, attenuation, out);
                            break;
                        }
                        //CosinePDF用于材质表面采样
                        const CosinePDF cosinePDF[] = {
                                CosinePDF(record.normalVector)
                        };

                        //将HittablePDF和CosinePDF组合进MixturePDF
                        HittablePDF hittablePDF[32] {};
                        size_t hittablePDFCount = 0;

                        for (size_t i = 0; i < dev_renderer->hittablePDFSphereCount; i++) {
                            hittablePDF[hittablePDFCount++] = HittablePDF(PrimitiveType::SPHERE, i, record.hitPoint);
                        }
                        for (size_t i = 0; i < dev_renderer->hittablePDFParallelogramCount; i++) {
                            hittablePDF[hittablePDFCount++] = HittablePDF(PrimitiveType::PARALLELOGRAM, i, record.hitPoint);
                        }

                        const MixturePDF pdf(cosinePDF, hittablePDF,
                                             1, dev_renderer->hittablePDFSphereCount + dev_renderer->hittablePDFParallelogramCount);

                        //使用MixturePDF生成一个新的光线方向
                        const Vec3 direction = pdf.generate(state, dev_renderer->dev_hittablePDFSpheres, dev_renderer->dev_hittablePDFParallelograms);
                        out = Ray(record.hitPoint, direction, ray->time);
                        const double pdfValue = pdf.value(dev_renderer->dev_hittablePDFSpheres, dev_renderer->dev_hittablePDFParallelograms, out.direction);

                        //pdfValue有效性检查
                        if (isnan(pdfValue) || isinf(pdfValue) || floatValueNearZero(pdfValue)) {
                            return Color3();
                        }

                        const Color3 BRDFvalue = dev_renderer->dev_roughMaterials[record.materialIndex].evalBRDF(currentRay, record);
                        const double cosTheta = dev_renderer->dev_roughMaterials[record.materialIndex].cosTheta(out, record);
                        result *= BRDFvalue * cosTheta / pdfValue;

                        attenuation = BRDFvalue * PI;
                        break;
                    }
                    case MaterialType::METAL:
                        if (!dev_renderer->dev_metalMaterials[record.materialIndex].scatter(state, currentRay, record, attenuation, out)) {
                            return result;
                        }
                        break;
                    case MaterialType::DIELECTRIC:
                        dev_renderer->dev_dielectricMaterials[record.materialIndex].scatter(state, currentRay, record, attenuation, out);
                        break;
                    default:;
                }

                currentRay = out;
                result *= attenuation; //光线衰减系数
            } else {
                result *= dev_cam->backgroundColor; //没有发生碰撞，将背景光颜色作为光源乘入结果并结束追踪循环
                break;
            }
        }
        return result;
    }

    __device__ Ray constructRay(const Camera * dev_cam, const Point3 & samplePoint, curandState * state) {
        //离焦采样：在离焦半径内随机选取一个点，以这个点发射光线
        Point3 rayOrigin = dev_cam->cameraCenter;
        if (dev_cam->focusDiskRadius > 0.0) {
            const Vec3 defocusVector = Vec3::randomPlaneVectorDevice(state, dev_cam->focusDiskRadius);
            //使用视口方向向量定位采样点
            rayOrigin = dev_cam->cameraCenter + defocusVector[0] * dev_cam->cameraU + defocusVector[1] * dev_cam->cameraV;
        }

        //在快门开启时段内随机找一个时刻发射光线
        const Vec3 rayDirection = Point3::constructVector(rayOrigin, samplePoint).unitVector();
        return Ray(rayOrigin, rayDirection, randomDoubleDevice(state, dev_cam->shutterRange.min, dev_cam->shutterRange.max));
    }

    /*
     * 初始化设备端随机数生成器
     * 渲染开始前为每一个线程初始化curandState对象，并存储在全局内存中
     * 在渲染函数中通过线程id访问curandState对象
     */
    __global__ void initThreadRandom(curandState * dev_stateArray) {
        const Uint32 x = blockIdx.x * blockDim.x + threadIdx.x;
        const Uint32 y = blockIdx.y * blockDim.y + threadIdx.y;
        const Uint32 tid = gridDim.x * blockDim.x * y + x;

        //使用线程id和处理器时间合成随机数种子
        curand_init(tid ^ clock64(), tid, 0, dev_stateArray + tid);
    }

    __global__ void render(const Renderer * dev_renderer, const Camera * dev_cam, Uint32 * dev_pixelBuffer, curandState * dev_stateArray) {
        //当前线程对应的全局像素坐标
        const Uint32 x = blockIdx.x * blockDim.x + threadIdx.x;
        const Uint32 y = blockIdx.y * blockDim.y + threadIdx.y;
        const Uint32 pixelIndex = gridDim.x * blockDim.x * y + x;

        if (x >= dev_cam->windowWidth || y >= dev_cam->windowHeight) return;

        curandState * threadState = dev_stateArray + pixelIndex;
        Color3 result;

        //抗锯齿采样
        for (size_t sampleI = 0; sampleI < dev_cam->sqrtSampleCount; sampleI++) {
            for (size_t sampleJ = 0; sampleJ < dev_cam->sqrtSampleCount; sampleJ++) {
                const double offsetX = ((static_cast<double>(sampleJ) + randomDoubleDevice(threadState)) * dev_cam->reciprocalSqrtSampleCount) - 0.5;
                const double offsetY = ((static_cast<double>(sampleI) + randomDoubleDevice(threadState)) * dev_cam->reciprocalSqrtSampleCount) - 0.5;
                const Point3 samplePoint =
                        dev_cam->pixelOrigin + ((x + offsetX) * dev_cam->viewPortPixelDx) + ((y + offsetY) * dev_cam->viewPortPixelDy);

                //构造光线
                const Ray ray = constructRay(dev_cam, samplePoint, threadState);

                //发射光线
                //const size_t sampleIndex = sampleI * dev_cam->sqrtSampleCount + sampleJ;
                result += rayColor(dev_renderer, dev_cam, &ray, threadState);
            }
        }

        //取平均值
        result *= dev_cam->reciprocalSqrtSampleCount * dev_cam->reciprocalSqrtSampleCount;

        //写入到缓冲区
        result.writeColor((Uint8*)(dev_pixelBuffer + pixelIndex));
    }
}