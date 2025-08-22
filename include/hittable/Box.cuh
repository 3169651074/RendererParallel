#ifndef RENDERERPARALLEL_BOX_CUH
#define RENDERERPARALLEL_BOX_CUH

#include <hittable/Parallelogram.cuh>

namespace renderer {
    class Box {
    public:
        //使用两个对角点定义Box
        Point3 min, max;

        //材质属性
        MaterialType materialType;
        size_t materialIndex;

        bool isTransformed;

        __host__ Box(MaterialType matType, size_t matIndex, const Point3 & p1, const Point3 & p2, bool isTransformed = false) :
                materialType(matType), materialIndex(matIndex), isTransformed(isTransformed) {
            //找出p1和p2的大小关系
            for (int i = 0; i < 3; i++) {
                min[i] = std::min(p1[i], p2[i]);
                max[i] = std::max(p1[i], p2[i]);
            }
        }

        __host__ BoundingBox constructBoundingBox() const {
            return {min, max};
        }

        __host__ Point3 centroid() const {
            const Vec3 half = Point3::constructVector(min, max);
            return min + 0.5 * half;
        }

        //使用AABB同款Slab-Test方法进行碰撞检测
        __device__ bool hit(const Ray & ray, const Range & range, HitRecord & hitInfo) const {
            double t_min = range.min;
            double t_max = range.max;

            for (int i = 0; i < 3; i++) {
                const double invD = 1.0 / ray.direction[i];
                double t0 = (min[i] - ray.origin[i]) * invD;
                double t1 = (max[i] - ray.origin[i]) * invD;

                //如果光线方向为负，t0和t1会交换大小，这里确保 t0 是与较小平面相交的 t 值
                if (invD < 0.0) {
                    //std::swap(t0, t1);
                    const double tmp = t0;
                    t0 = t1;
                    t1 = tmp;
                }

                //更新总的 t 区间
                //t_min = max(t_min, t0);
                //t_max = min(t_max, t1);
                if (t0 > t_min) t_min = t0;
                if (t1 < t_max) t_max = t1;

                //如果区间不重叠，则不可能命中
                if (t_min >= t_max) {
                    return false;
                }
            }

            //此时，t_min 是光线进入盒子的时间点，t_max 是离开的时间点
            //我们只关心最近的有效撞击点
            if (!range.inRange(t_min)) {
                //如果 t_min 不在有效范围内，可能 t_max 在（光线从盒子内部发出）
                //但对于标准的光线追踪，我们通常只关心第一个交点
                //这里可以根据需求决定是否测试 t_max
                return false;
            }

            // 记录碰撞信息
            hitInfo.t = t_min;
            hitInfo.hitPoint = ray.at(t_min);
            hitInfo.materialType = materialType;
            hitInfo.materialIndex = materialIndex;

            //计算法向量
            //找出撞击点在哪一个面上
            Vec3 outwardNormal;
            if (abs(hitInfo.hitPoint[0] - min[0]) < FLOAT_VALUE_ZERO_EPSILON) {
                outwardNormal = Vec3(-1, 0, 0);
            } else if (abs(hitInfo.hitPoint[0] - max[0]) < FLOAT_VALUE_ZERO_EPSILON) {
                outwardNormal = Vec3(1, 0, 0);
            } else if (abs(hitInfo.hitPoint[1] - min[1]) < FLOAT_VALUE_ZERO_EPSILON) {
                outwardNormal = Vec3(0, -1, 0);
            } else if (abs(hitInfo.hitPoint[1] - max[1]) < FLOAT_VALUE_ZERO_EPSILON) {
                outwardNormal = Vec3(0, 1, 0);
            } else if (abs(hitInfo.hitPoint[2] - min[2]) < FLOAT_VALUE_ZERO_EPSILON) {
                outwardNormal = Vec3(0, 0, -1);
            } else {
                outwardNormal = Vec3(0, 0, 1);
            }
            hitInfo.normalVector = outwardNormal;
            hitInfo.hitFrontFace = Vec3::dot(ray.direction, outwardNormal) < 0.0;

            //省略UV坐标的映射
            return true;
        }
    };
}

#endif //RENDERERPARALLEL_BOX_CUH
