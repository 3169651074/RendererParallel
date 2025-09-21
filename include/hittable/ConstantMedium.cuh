#ifndef RENDERERPARALLEL_CONSTANTMEDIUM_CUH
#define RENDERERPARALLEL_CONSTANTMEDIUM_CUH

#include <box/BVHTree.cuh>

namespace renderer {
    /*
     * 同时包装了物体和材质
     */
    class ConstantMedium {
    public:
        //物体索引
        PrimitiveType primitiveType;
        size_t primitiveIndex;

        //物体包围盒和中心点
        BoundingBox boundingBox;
        Point3 objectCentroid;

        //材质索引
        MaterialType materialType;
        size_t materialIndex;

        //密度
        double factor;

        //均匀介质类可以包装变换类，本身不能被变换
        bool isTransformed = false;

        __host__ ConstantMedium(PrimitiveType primitiveType, size_t primitiveIndex, const BoundingBox & boundingBox, const Point3 & centroid, MaterialType materialType, size_t materialIndex, double density) :
            primitiveType(primitiveType), primitiveIndex(primitiveIndex), boundingBox(boundingBox), objectCentroid(centroid), materialType(materialType), materialIndex(materialIndex), factor(-1.0 / density) {}

        //ConstantMedium和Transform都是包装类，hit函数需要传入所有类型的图元列表供索引
        __device__ bool hit(curandState * state, const Ray & ray, const Range & range, HitRecord & record,
                            const Sphere * spheres,
                            const Triangle * triangles,
                            const Parallelogram * parallelograms,
                            const Box * boxes,
                            const Transform * transforms) const
        {
            /*
             * 首先调用边界物体的 hit 函数两次，以找到光线进入和射出该体积的两个交点
             * 如果光线没有进入或者只进入一次（擦边），则认为没有命中
             * 每次碰撞都随机变换方向，多次迭代后就能形成柔和、弥散的视觉效果
             *
             * 1：正常与物体求交，找到一个交点（hit方法自动记录最近交点）
             * 2：前进一小段距离，再次求交，尝试寻找第二个交点
             * 3：光线在介质中行进的距离为两个交点的t值之差乘以光线的方向向量的模
             */
            HitRecord rec1, rec2;
            switch (primitiveType) {
#define _primitiveHitTest1(arrayName, typeName)\
                case PrimitiveType::typeName:\
                    if (!arrayName[primitiveIndex].hit(ray, Range(-INFINITY, INFINITY), rec1)) return false;\
                    break

                _primitiveHitTest1(spheres, SPHERE);
                _primitiveHitTest1(triangles, TRIANGLE);
                _primitiveHitTest1(parallelograms, PARALLELOGRAM);
                _primitiveHitTest1(boxes, BOX);
#undef _primitiveHitTest1

                case PrimitiveType::TRANSFORM:
                    if (!transforms[primitiveIndex].hit(ray, Range(-INFINITY, INFINITY), rec1,
                                                        spheres, triangles, parallelograms, boxes)) return false;
                    break;
                default:;
            }

            switch (primitiveType) {
#define _primitiveHitTest2(arrayName, typeName)\
                case PrimitiveType::typeName:\
                    if (!arrayName[primitiveIndex].hit(ray, Range(rec1.t + 0.001, INFINITY), rec2)) return false;\
                    break

                _primitiveHitTest2(spheres, SPHERE);
                _primitiveHitTest2(triangles, TRIANGLE);
                _primitiveHitTest2(parallelograms, PARALLELOGRAM);
                _primitiveHitTest2(boxes, BOX);
#undef _primitiveHitTest2

                case PrimitiveType::TRANSFORM:
                    if (!transforms[primitiveIndex].hit(ray, Range(rec1.t + 0.001, INFINITY), rec2,
                                                        spheres, triangles, parallelograms, boxes)) return false;
                    break;
                default:;
            }

            if (rec1.t < range.min) {
                rec1.t = range.min;
            }
            if (rec2.t > range.max) {
                rec2.t = range.max;
            }

            if (rec1.t >= rec2.t) {
                return false;
            }
            if (rec1.t < 0) {
                rec1.t = 0;
            }

            //计算出光线在介质内部需要穿行的总距离
            //确保无论光线是否被变换过，距离计算都准确
            const auto rayLength = ray.direction.length();
            const auto distanceInsideBoundary = (rec2.t - rec1.t) * rayLength;

            /*
             * 根据介质的密度决定光在撞上一个第一个粒子前能走多远
             * 使用比尔-朗伯定律
             * hitDistance = −1 / density × ln(randomDouble())
             * 密度越大，平均的hitDistance就越短
             */
            const auto hitDistance = factor * log(randomDoubleDevice(state));

            if (hitDistance > distanceInsideBoundary) {
                //光没有在介质内部发生散射
                return false;
            } else {
                //发生散射，计算碰撞点并填充碰撞信息
                record.t = rec1.t + hitDistance / rayLength;
                record.hitPoint = ray.at(record.t);
                record.normalVector = Vec3(1.0, 0.0, 0.0);
                record.hitFrontFace = true;
                record.materialType = materialType;
                record.materialIndex = materialIndex;
                return true;
            }
        }

        __host__ BoundingBox constructBoundingBox() const {
            return boundingBox;
        }

        __host__ Point3 centroid() const {
            return objectCentroid;
        }
    };
}

#endif //RENDERERPARALLEL_CONSTANTMEDIUM_CUH
