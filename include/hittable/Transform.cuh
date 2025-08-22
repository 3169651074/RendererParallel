#ifndef RENDERERPARALLEL_TRANSFORM_CUH
#define RENDERERPARALLEL_TRANSFORM_CUH

#include <box/BoundingBox.cuh>
#include <hittable/Sphere.cuh>
#include <hittable/Triangle.cuh>
#include <hittable/Parallelogram.cuh>
#include <hittable/Box.cuh>

namespace renderer {
    /*
     * Transform::hit需要被GPU线程调用，但是hit方法的调用次数未知
     * 如果在每次hit调用都在全局显存中分配临时变量，会导致大量显存碎片，导致无法分配大块连续显存
     * hit方法内部所有的临时变量（如变换后的光线起点、方向等）都必须是本地栈变量，存储在每个线程私有的寄存器内
     */
    class Transform {
    public:
        //变换的物体类型和索引
        PrimitiveType primitiveType;
        size_t primitiveIndex;

        //变换矩阵
        Matrix transformMatrix;
        Matrix transformInverse;
        Matrix transformInverseTranspose;

        //变换后物体的包围盒和中心点
        BoundingBox transformedBoundingBox;
        Point3 transformedCentroid;

        __host__ Transform(PrimitiveType primitiveType, size_t primitiveIndex, const BoundingBox & boundingBox, const Point3 & centroid,
                const std::array<double, 3> & rotate = {}, const std::array<double, 3> & shift = {}, const std::array<double, 3> & scale = {1.0, 1.0, 1.0}) :
                primitiveType(primitiveType), primitiveIndex(primitiveIndex)
        {
            //M = T * R * S，平移 * 旋转 * 缩放
            const auto m1 = Matrix::constructShiftMatrix(shift);
            const auto m2 = Matrix::constructRotateMatrix(rotate);
            const auto m3 = Matrix::constructScaleMatrix(scale);
            transformMatrix = m1 * m2 * m3;
            transformInverse = transformMatrix.inverse();
            transformInverseTranspose = transformInverse.transpose();

            //变换包围盒和中心点
            this->transformedBoundingBox = boundingBox.transformBoundingBox(transformMatrix);
            this->transformedCentroid = Matrix::toPoint(transformMatrix * Matrix::toMatrix(centroid));
        }

        /*
         * Transform类为一个hittable，同时也是一个托管，可能包含任何图元
         * 函数需要所有类型的图元列表
         */
        __device__ bool hit(const Ray & ray, const Range & range, HitRecord & record,
                            const Sphere * spheres,
                            const Triangle * triangles,
                            const Parallelogram * parallelograms,
                            const Box * boxes) const
        {
            /*
             * 将世界空间光线变换到物体的局部空间：使用逆矩阵分别对ray的起点和方向向量进行变换
             * 只有左矩阵的列数和右矩阵的行数相同的矩阵才能相乘，则将三维点变为1列4行的列向量
             */
            auto rayOrigin = Matrix::toMatrix(ray.origin);
            auto rayDirection = Matrix::toMatrix(ray.direction);
            rayOrigin = transformInverse * rayOrigin;
            rayDirection = transformInverse * rayDirection;

            //构造变换后的光线
            const Ray transformed(Matrix::toPoint(rayOrigin), Matrix::toVector(rayDirection), ray.time);

            //在物体空间中对变换后的光线进行相交测试
            bool isHit = false;
            switch (primitiveType) {
#define _primitiveHitTest(arrayName, typeName)\
                case PrimitiveType::typeName: \
                    if (arrayName[primitiveIndex].hit(transformed, range, record)) { isHit = true; }\
                    break;
                //============
                _primitiveHitTest(spheres, SPHERE);
                _primitiveHitTest(triangles, TRIANGLE);
                _primitiveHitTest(parallelograms, PARALLELOGRAM);
                _primitiveHitTest(boxes, BOX);
                //============
#undef _primitiveHitTest
                default:;
            }

            if (!isHit) {
                return false;
            } else {
                //如果有碰撞，则将将局部空间的命中记录变换回世界空间，t值和uv坐标不需要变换
                //变换碰撞点
                auto point = Matrix::toMatrix(record.hitPoint);
                point = transformMatrix * point;
                record.hitPoint = Matrix::toPoint(point);

                //使用逆转置变换矩阵变换法向量
                //向量不受平移影响，将w分量设置为0可完成此目的
                auto normal = Matrix::toMatrix(record.normalVector);
                normal = transformInverseTranspose * normal;
                record.normalVector = Matrix::toVector(normal).unitVector();
                record.hitFrontFace = Vec3::dot(ray.direction, record.normalVector) < 0.0;
                return true;
            }
        }

        __host__ BoundingBox constructBoundingBox() const {
            return transformedBoundingBox;
        }

        __host__ Point3 centroid() const {
            return transformedCentroid;
        }
    };
}

#endif //RENDERERPARALLEL_TRANSFORM_CUH
