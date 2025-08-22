#ifndef RENDERERPARALLEL_PARALLELOGRAM_CUH
#define RENDERERPARALLEL_PARALLELOGRAM_CUH

#include <box/BoundingBox.cuh>

namespace renderer {
    class Parallelogram {
    public:
        //物体属性
        Point3 q;
        Vec3 u, v;
        double area;

        Vec3 normalVector;
        double planeD;

        //材质属性
        MaterialType materialType;
        size_t materialIndex;

        bool isTransformed;

        // ====== 主机函数 ======

        __host__ Parallelogram(MaterialType materialType, size_t materialIndex,
               const Point3 & q, const Vec3 & u, const Vec3 & v, bool isTransformed = false) :
               materialType(materialType), materialIndex(materialIndex), q(q), u(u), v(v), isTransformed(isTransformed)
        {
            this->normalVector = Vec3::cross(u, v);
            this->area = normalVector.length();
            this->normalVector.unitize();
            double sum = 0.0;
            for (int i = 0; i < 3; i++) {
                sum += normalVector[i] * q[i];
            }
            this->planeD = sum;
        }

        __host__ BoundingBox constructBoundingBox() const {
            //将四个顶点都包进包围盒中
            BoundingBox b1(q, q + u + v), b2(q + u, q + v);
            return {b1, b2};
        }

        //获取四边形的中心点
        __host__ Point3 centroid() const {
            return q + 0.5 * u + 0.5 * v;
        }

        // ====== 设备函数 ======

        __device__ bool hit(const Ray & ray, const Range & range, HitRecord & hitInfo) const {
            const double NDotD = Vec3::dot(normalVector, ray.direction);
            if (floatValueNearZero(NDotD)) {
                return false;
            }

            //计算光线和四边形所在无限平面的交点参数t
            double NDotP = 0.0;
            for (int i = 0; i < 3; i++) {
                NDotP += normalVector[i] * ray.origin[i];
            }
            const double t = (planeD - NDotP) / NDotD;
            if (!range.inRange(t)) {
                return false;
            }

            //计算用四边形边向量表示的交点的系数，判断两个系数是否在[0, 1]范围内
            const Point3 intersection = ray.at(t);
            const Vec3 p = Point3::constructVector(q, intersection);
            const Vec3 normal = Vec3::cross(u, v);
            const double denominator = normal.lengthSquare();

            if (floatValueNearZero(denominator)) {
                return false;
            }

            const double alpha = Vec3::dot(Vec3::cross(p, v), normal) / denominator;
            const double beta = Vec3::dot(Vec3::cross(u, p), normal) / denominator;

            const Range coefficientRange(0.0, 1.0);
            if (!coefficientRange.inRange(alpha) || !coefficientRange.inRange(beta)) {
                return false;
            }

            //记录碰撞信息
            hitInfo.t = t;
            hitInfo.hitPoint = intersection;
            hitInfo.materialType = materialType;
            hitInfo.materialIndex = materialIndex;
            hitInfo.uvPair = Pair<double, double>(alpha, beta);
            hitInfo.hitFrontFace = Vec3::dot(ray.direction, normalVector) < 0.0;
            hitInfo.normalVector = hitInfo.hitFrontFace ? normalVector : -normalVector;
            return true;
        }

        __device__ double pdfValue(const Point3 &origin, const Vec3 &direction) const {
            HitRecord record;
            //检查方向有效性，确保从origin沿direction方向能够直接指向光源
            if (!this->hit(Ray(origin, direction), Range(0.001, INFINITY), record)) {
                return 0.0;
            }

            //从origin到q（光源上随机点）的向量为 record.t * direction
            const double distanceSquare = (record.t * direction).lengthSquare();
            //向量点积公式：cos(theta) = a dot b / |a| |b|，其中|b| = 1
            const double cosine = abs(Vec3::dot(direction, record.normalVector) / direction.length());
            return distanceSquare / (cosine * area);
        }

        __device__ Vec3 randomVector(curandState * state, const Point3 &origin) const {
            const Point3 to = q + (randomDoubleDevice(state) * u) + (randomDoubleDevice(state) * v);
            return Point3::constructVector(origin, to);
        }
    };
}

#endif //RENDERERPARALLEL_PARALLELOGRAM_CUH
