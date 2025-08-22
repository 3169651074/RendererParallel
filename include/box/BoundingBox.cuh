#ifndef RENDERERPARALLEL_BOUNDINGBOX_CUH
#define RENDERERPARALLEL_BOUNDINGBOX_CUH

#include <basic/Ray.cuh>
#include <util/Range.cuh>
#include <util/Matrix.cuh>

namespace renderer {
    /*
     * 轴对齐包围盒
     * 包围盒的构造由CPU完成，所有构造相关的函数都为__host__
     * 包围盒的相交测试由GPU线程完成
     */
    class BoundingBox {
    private:
        Range range[3];

        //确保包围盒体积有效
        __host__ void ensureVolume() {
            constexpr double EPSILON = FLOAT_VALUE_ZERO_EPSILON;
            for (auto & i : range) {
                if (i.length() < EPSILON) { i.expand(EPSILON); }
            }
        }

    public:
        //默认构造空包围盒
        __host__ explicit BoundingBox(const Range & x = Range(), const Range & y = Range(), const Range & z = Range()) {
            range[0] = x; range[1] = y; range[2] = z;
            ensureVolume();
        }

        //使用两个对角点构造长方体
        __host__ BoundingBox(const Point3 & p1, const Point3 & p2) {
            //取两个点每个分量的有效值
            for (size_t i = 0; i < 3; i++) {
                range[i] = p1[i] < p2[i] ? Range(p1[i], p2[i]) : Range(p2[i], p1[i]);
            }
            ensureVolume();
        }

        //使用bounds数组构造包围盒
        //[x1, x2], [y1, y2], [z1, z2]
        __host__ explicit BoundingBox(const double bounds[6]) {
            for (size_t i = 0; i < 6; i += 2) {
                range[i / 2] = Range(bounds[i], bounds[i + 1]);
            }
        }

        //构造两个包围盒的合并
        __host__ BoundingBox(const BoundingBox & b1, const BoundingBox & b2) {
            for (size_t i = 0; i < 3; i++) {
                range[i] = Range(b1.range[i], b2.range[i]);
            }
            //合并不会减小包围盒的体积
        }

        //使用矩阵变换包围盒
        __host__ BoundingBox transformBoundingBox(const Matrix & matrix) const {
            //使用矩阵对包围盒的8个顶点进行变换
            Point3 min(INFINITY, INFINITY, INFINITY);
            Point3 max(-INFINITY, -INFINITY, -INFINITY);

            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    for (int k = 0; k < 2; k++) {
                        //取出每个顶点的坐标
                        const double x = i * range[0].max + (1.0 - i) * range[0].min;
                        const double y = j * range[1].max + (1.0 - j) * range[1].min;
                        const double z = k * range[2].max + (1.0 - k) * range[2].min;

                        //计算变换后的坐标
                        const auto matrixPoint = Matrix::toMatrix(Point3(x, y, z));
                        const auto mul = matrix * matrixPoint;
                        const auto point = Matrix::toPoint(mul);

                        //计算最值，保证包围盒和坐标轴对齐
                        for (int l = 0; l < 3; l++) {
                            min[l] = std::min(min[l], point[l]);
                            max[l] = std::max(max[l], point[l]);
                        }
                    }
                }
            }
            //使用min和max重构包围盒
            return {min, max};
        }

        //包围盒求交函数，设备函数
        __device__ bool hit(const Ray & ray, const Range & checkRange, double & t) const {
            const Point3 & rayOrigin = ray.origin;
            const Vec3 & rayDirection = ray.direction;

            Range currentRange(checkRange);
            for (Uint32 axis = 0; axis < 3; axis++) {
                const Range & axisRange = range[axis];
                const double q = rayOrigin[axis];
                const double d = rayDirection[axis];

                //光线和包围盒平行
                if (std::abs(d) < FLOAT_VALUE_ZERO_EPSILON) {
                    //光线起点不在包围盒内，没有交点
                    if (q < axisRange.min || q > axisRange.max) return false;
                    //光线从包围盒内部发出，继续测试下一个轴
                    continue;
                }

                //计算光在当前轴和边界的两个交点
                double t1 = (axisRange.min - q) / d;
                double t2 = (axisRange.max - q) / d;

                //将currentRange限制到这两个交点的范围内
                if (t1 < t2) {
                    if (t1 > currentRange.min) currentRange.min = t1;
                    if (t2 < currentRange.max) currentRange.max = t2;
                } else {
                    if (t2 > currentRange.min) currentRange.min = t2;
                    if (t1 < currentRange.max) currentRange.max = t1;
                }

                if (!currentRange.isValid()) {
                    return false;
                }
            }

            t = currentRange.min;
            return true;
        }
    };
}

#endif //RENDERERPARALLEL_BOUNDINGBOX_CUH
