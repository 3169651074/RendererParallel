#ifndef RENDERERPARALLEL_STRUCTS_CUH
#define RENDERERPARALLEL_STRUCTS_CUH

#include <basic/Point3.cuh>
#include <util/Pair.cuh>

namespace renderer {
    //图元类型枚举
    typedef enum class PrimitiveType {
        SPHERE, TRIANGLE, PARALLELOGRAM, BOX, TRANSFORM
    } PrimitiveType;

    //材质类型枚举
    typedef enum class MaterialType {
        ROUGH, METAL, DIELECTRIC, DIFFUSE_LIGHT
    } MaterialType;

    //PDF类型枚举
    typedef enum class PDFType {
        COSINE, HITTABLE, MIXTURE
    } PDFType;

    //碰撞信息
    typedef struct HitRecord {
        Point3 hitPoint;
        Vec3 normalVector;
        double t;
        bool hitFrontFace;

        /*
         * 使用材质类型和材质索引（对应一种材质的不同对象）代替材质指针
         * 两个变量结合定位具体材质对象
         */
        MaterialType materialType;
        size_t materialIndex;

        Pair<double, double> uvPair;

        //通用POD结构体不需要显式提供构造等成员函数
    } HitRecord;
}

#endif //RENDERERPARALLEL_STRUCTS_CUH
