#ifndef RENDERERPARALLEL_TRIANGLE_CUH
#define RENDERERPARALLEL_TRIANGLE_CUH

#include <box/BoundingBox.cuh>

namespace renderer {
    /*
     * 平面三角形类
     */
    class Triangle {
    public:
        //物体属性
        Point3 points[3];
        Vec3 normalVector[3];
        Vec3 e1, e2;
        Vec3 direction; //从运动起点指向运动终点的向量

        //材质属性
        MaterialType materialType;
        size_t materialIndex;

        bool isTransformed;

        //使用三个顶点构造静止三角形，面法向量垂直于三角形平面
        __host__ Triangle(MaterialType materialType, size_t materialIndex,
                 const Point3 & p1, const Point3 & p2, const Point3 & p3, bool isTransformed = false) :
                materialType(materialType), materialIndex(materialIndex), direction(Vec3()), isTransformed(isTransformed)
        {
            points[0] = p1; points[1] = p2; points[2] = p3;
            e1 = Point3::constructVector(p1, p2);
            e2 = Point3::constructVector(p1, p3);
            for (auto & i : normalVector) {
                i = Vec3::cross(e1, e2).unitVector();
            }
        }

        //使用顶点（三个起点和第一个顶点的终点）和独立的顶点法向量构造运动三角形（直线运动）
        __host__ Triangle(MaterialType materialType, size_t materialIndex,
                 const Point3 & p1, const Point3 & p2, const Point3 & p3,
                 const Vec3 & normal1, const Vec3 & normal2, const Vec3 & normal3, const Vec3 & direction, bool isTransformed = false) :
                materialType(materialType), materialIndex(materialIndex), direction(direction), isTransformed(isTransformed)
        {
            points[0] = p1; points[1] = p2; points[2] = p3;
            normalVector[0] = normal1; normalVector[1] = normal2; normalVector[2] = normal3;
            e1 = Point3::constructVector(p1, p2);
            e2 = Point3::constructVector(p1, p3);
        }

        //拷贝构造，赋值运算符和析构函数使用默认版本

        //构造包围盒
        __host__ BoundingBox constructBoundingBox() const {
            //找出运动位置起终点顶点每个分量的最小值和最大值
            const Point3 p1End = points[0] + direction;
            const Point3 p2End = points[1] + direction;
            const Point3 p3End = points[2] + direction;

            const Point3 minPoint(
                    std::min({points[0][0], points[1][0], points[2][0], p1End[0], p2End[0], p3End[0]}),
                    std::min({points[0][1], points[1][1], points[2][1], p1End[1], p2End[1], p3End[1]}),
                    std::min({points[0][2], points[1][2], points[2][2], p1End[2], p2End[2], p3End[2]})
            );
            const Point3 maxPoint(
                    std::max({points[0][0], points[1][0], points[2][0], p1End[0], p2End[0], p3End[0]}),
                    std::max({points[0][1], points[1][1], points[2][1], p1End[1], p2End[1], p3End[1]}),
                    std::max({points[0][2], points[1][2], points[2][2], p1End[2], p2End[2], p3End[2]})
            );
            return {minPoint, maxPoint};
        }

        //计算重心：三个顶点坐标的平均值
        __host__ Point3 centroid() const {
            Point3 ret;
            for (size_t i = 0; i < 3; i++) {
                ret[i] = (points[0][i] + points[1][i] + points[2][i]) / 3.0;
            }
            return ret;
        }

        //碰撞检测
        __device__ bool hit(const Ray & ray, const Range & range, HitRecord & record) const {
            const Vec3 h = ray.direction.cross(e2); //h = d x e2
            //系数行列式
            const double detA = e1.dot(h); //detA = e1 * (d x e2)

            //行列式为0，说明方程组无解或有无穷解（光线和三角形平行或有无数个交点）
            if (floatValueNearZero(detA)) {
                return false;
            }

            const Vec3 s = Point3::constructVector(points[0], ray.origin); //s = O - v0

            //计算未知数U并检查
            const Range coefficientRange(0.0, 1.0);
            const double u = s.dot(h) / detA; // u = (s · h) / det
            if (!coefficientRange.inRange(u)) {
                return false;
            }

            const Vec3 q = s.cross(e1);  // q = s × e1

            //计算未知数V并检查
            const double v = ray.direction.dot(q) / detA; // v = (D · q) / det
            if (!coefficientRange.inRange(v) || u + v > 1.0) {
                return false;
            }

            //满足相交条件，计算碰撞参数
            record.t = e2.dot(q) / detA; // t = (e2 · q) / det
            if (!range.inRange(record.t)) {
                return false;
            }
            record.hitPoint = ray.at(record.t);
            record.materialType = materialType;
            record.materialIndex = materialIndex;
            record.uvPair = Pair<double, double>(u, v);

            //交点法向量为三个顶点法向量的插值平滑
            const Vec3 n = ((1.0 - u - v) * normalVector[0] + u * normalVector[1] + v * normalVector[2]).unitVector();
            record.hitFrontFace = Vec3::dot(ray.direction, n) < 0.0;
            record.normalVector = record.hitFrontFace ? n : -n;
            return true;
        }
    };
}

#endif //RENDERERPARALLEL_TRIANGLE_CUH
