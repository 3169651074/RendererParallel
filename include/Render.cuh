#ifndef RENDERERPARALLEL_RENDER_CUH
#define RENDERERPARALLEL_RENDER_CUH

#include <Camera.cuh>
#include <box/BVHTree.cuh>
#include <material/Rough.cuh>
#include <material/Metal.cuh>
#include <material/Dielectric.cuh>
#include <material/DiffuseLight.cuh>
#include <pdf/MixturePDF.cuh>

namespace renderer {
    /*
     * 渲染器类，包含场景信息
     * 画面信息如摄像机信息在渲染时传入
     */
    class Renderer {
    public:
        //材质显存指针
        Rough * dev_roughMaterials = nullptr;
        Metal * dev_metalMaterials = nullptr;
        Dielectric * dev_dielectricMaterials = nullptr;
        DiffuseLight * dev_diffuseLightMaterials = nullptr;

        //图元显存指针
        Sphere * dev_spheres = nullptr;
        Triangle * dev_triangles = nullptr;
        Parallelogram * dev_parallelograms = nullptr;
        Box * dev_boxes = nullptr;
        Transform * dev_transforms = nullptr;

        //显存指针是否成功初始化
        bool devPointerAvailable = false;

        //直接重要性采样物体列表，指向显存中的指针数组
        /*
         * const Sphere ** 表示指向的Sphere对象不可变
         * Sphere * const * --> const修饰第一级指针，不能修改指针本身
         * Sphere ** const --> const修饰第二级指针，不能修改二级指针本身
         */
        const Sphere ** dev_hittablePDFSpheres = nullptr;
        size_t hittablePDFSphereCount = 0;
        const Parallelogram ** dev_hittablePDFParallelograms = nullptr;
        size_t hittablePDFParallelogramCount = 0;

        //是否启用重要性采样
        bool isDirectSample = false;

        //设置直接采样物体列表，传入图元类型和索引数组
        __host__ void setDirectSampleObject(const std::pair<PrimitiveType, size_t> * objectList, size_t objectListSize);

        //BVH信息，由场景中物体决定
        BVHTree::BVHTreeNode * dev_tree = nullptr;
        std::pair<PrimitiveType, size_t> * dev_treeIndexArray = nullptr;

        //分配显存并拷贝数据，传入所有场景信息，返回显存指针
        void commitSceneData(
                const Rough * roughMaterials, Uint32 roughMaterialCount,
                const Metal * metalMaterials, Uint32 metalMaterialCount,
                const Dielectric * dielectricMaterials, Uint32 dielectricMaterialCount,
                const DiffuseLight * diffuseLightMaterials, Uint32 diffuseLightMaterialCount,
                const Sphere * spheres, Uint32 sphereCount,
                const Triangle * triangles, Uint32 triangleCount,
                const Parallelogram * parallelograms, Uint32 parallelogramCount,
                const Box * boxs, Uint32 boxCount,
                const Transform * transforms, Uint32 transformCount);

        //释放场景显存
        void freeSceneData();

        //主渲染函数：传入和当前帧信息（相机对象），并将结果写入传入的窗口后更新画面
        void renderFrame(const Camera * cam, SDL_Window * window, bool isPrintInfo = true) const;

        //打印cuda设备信息
        static void printDeviceInfo();
    };

    // ====== 核函数，实现在Kernel.cu中 ======

    //初始化线程随机数生成器
    __global__ void initThreadRandom(curandState * dev_stateArray);

    //主渲染核函数
    __global__ void render(const Renderer * dev_renderer, const Camera * dev_cam, Uint32 * dev_pixelBuffer, curandState * dev_stateArray);
}

#endif //RENDERERPARALLEL_RENDER_CUH
