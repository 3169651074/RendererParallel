#ifndef RENDERERPARALLEL_SPHERE_CUH
#define RENDERERPARALLEL_SPHERE_CUH

#include <box/BoundingBox.cuh>
#include <util/OrthonormalBase.cuh>

namespace renderer {
    /*
     * 球体类，包围盒不在构造函数中计算，而是在构建BVH树时统一计算
     * 主机端和设备端通用，但只能在主机端构造
     */
    class Sphere {
    public:
        //物体属性
        Ray center;
        double radius;

        //材质属性
        MaterialType materialType;
        size_t materialIndex;

        //是否参与变换
        bool isTransformed;

        // ====== 主机函数 ======

        //构造静止球体
        __host__ Sphere(MaterialType materialType, size_t materialIndex, const Point3 & center, double radius, bool isTransformed = false) :
                materialType(materialType), materialIndex(materialIndex), center(Ray(center, Vec3())), radius(radius > 0.0 ? radius : 0.0), isTransformed(isTransformed) {}

        //构造运动球体
        __host__ Sphere(MaterialType materialType, size_t materialIndex, const Point3 & from, const Point3 & to, double radius, bool isTransformed = false) :
                materialType(materialType), materialIndex(materialIndex), center(Ray(from, Point3::constructVector(from, to))), radius(radius > 0.0 ? radius : 0.0), isTransformed(isTransformed)  {}

        //拷贝构造，赋值运算符和析构函数使用默认版本

        //构造包围盒
        __host__ BoundingBox constructBoundingBox() const {
            const Vec3 edge = Vec3(radius, radius, radius);
            if (center.direction.lengthSquare() < Vec3::VECTOR_LENGTH_SQUARE_ZERO_EPSILON) {
                return {center.origin - edge, center.origin + edge};
            } else {
                const Point3 to = center.origin + center.direction;
                auto bStart = BoundingBox(center.origin - edge, center.origin + edge);
                auto bEnd = BoundingBox(to - edge, to + edge);
                return {bStart, bEnd};
            }
        }

        __host__ Point3 centroid() const {
            return center.origin;
        }

        // ====== 设备函数 ======

        //纹理映射。将位于球体表面的点转换为二维坐标（u, v）
        __device__ static Pair<double, double> mapUVPair(const Point3 & surfacePoint) {
            const double theta = std::acos(-surfacePoint[1]);
            const double phi = std::atan2(-surfacePoint[2], surfacePoint[0]) + PI;

            Pair<double, double> ret;
            ret.first = phi / (2.0 * PI);
            ret.second = theta / PI;
            return ret;
        }

        //碰撞检测
        __device__ bool hit(const Ray & ray, const Range & range, HitRecord & record) const {
            //获取球体在当前时间的中心位置
            const Point3 currentCenter = center.at(ray.time);

            //解一元二次方程，判断光线和球体的交点个数
            const Vec3 cq = Point3::constructVector(ray.origin, currentCenter);
            const Vec3 dir = ray.direction;
            const double a = Vec3::dot(dir, dir);
            const double b = -2.0 * Vec3::dot(cq, dir);
            const double c = Vec3::dot(cq, cq) - radius * radius;
            double delta = b * b - 4.0 * a * c;

            if (delta < 0.0) return false;
            delta = sqrt(delta);

            //root1对应较小的t值，为距离摄像机较近的交点
            const double root1 = (-b - delta) / (a * 2.0);
            const double root2 = (-b + delta) / (a * 2.0);

            double root;
            if (range.inRange(root1)) { //先判断root1
                root = root1;
            } else if (range.inRange(root2)) {
                root = root2;
            } else {
                return false; //两个根均不在允许范围内
            }

            //设置碰撞信息
            record.t = root;
            record.hitPoint = ray.at(root);
            record.materialType = materialType;
            record.materialIndex = materialIndex;

            //outwardNormal为球面向外的单位法向量，通过此向量和光线方向向量的点积符号判断光线撞击了球的内表面还是外表面
            //若点积小于0，则两向量夹角大于90度，两向量不同方向
            const Vec3 outwardNormal = Point3::constructVector(currentCenter, record.hitPoint).unitVector();
            record.hitFrontFace = Vec3::dot(ray.direction, outwardNormal) < 0;
            record.normalVector = record.hitFrontFace ? outwardNormal : -outwardNormal;

            //将碰撞点从世界坐标系变换到以球心为原点的局部坐标系：直接在世界坐标系中构造向量
            const Vec3 localVector = Point3::constructVector(currentCenter, record.hitPoint).unitVector();
            record.uvPair = mapUVPair(Point3(localVector));
            return true;
        }

        __device__ double pdfValue(const Point3 &origin, const Vec3 &direction) const {
            //此计算方法只对静止球体有效
            HitRecord record;
            if (!this->hit(Ray(origin, direction), Range(0.001, INFINITY), record)) {
                return 0.0;
            }

            const double distanceSquare = Point3::distanceSquare(origin, center.at(0.0));
            const double cosThetaMax = sqrt(1.0 - radius * radius / distanceSquare);
            const double solidAngle = 2.0 * PI * (1.0 - cosThetaMax);
            return 1.0 / solidAngle;
        }

        __device__ Vec3 randomVector(curandState * state, const Point3 &origin) const {
            //此计算方法只对静止球体有效
            const Vec3 direction = Point3::constructVector(origin, center.at(0.0));
            const double distanceSquare = direction.lengthSquare();

            const double r1 = randomDoubleDevice(state);
            const double r2 = randomDoubleDevice(state);

            const double phi = 2.0 * PI * r1;
            const double z = 1.0 + r2 * (sqrt(1.0 - radius * radius / distanceSquare) - 1);
            const double x = cos(phi) * sqrt(1.0 - z * z);
            const double y = sin(phi) * sqrt(1.0 - z * z);

            const OrthonormalBase base(direction, 2);
            return base.transformToWorld(Vec3(x, y, z));
        }
    };
}

#endif //RENDERERPARALLEL_SPHERE_CUH
