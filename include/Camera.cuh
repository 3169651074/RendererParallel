#ifndef RENDERERPARALLEL_CAMERA_CUH
#define RENDERERPARALLEL_CAMERA_CUH

#include <basic/Ray.cuh>
#include <basic/Color3.cuh>

namespace renderer {
    /*
     * 相机类
     */
    class Camera {
    public:
        // ====== 窗口属性 ======
        Uint32 windowWidth;
        Uint32 windowHeight;

        //默认背景颜色，填充没有物体的区域
        Color3 backgroundColor;

        // ====== 相机属性 ======
        Point3 cameraCenter;
        Point3 cameraTarget;                    //相机放置在cameraCenter，看向cameraTarget，此两点间距为焦距
        double horizontalFOV;                   //水平方向的视角，构造时单位为角度，决定视口宽度

        //视口属性
        double viewPortWidth;
        double viewPortHeight;
        Vec3 upDirection;

        /*
         * 视口平面的基向量，用于定位视口
         * U指向相机的右侧（视口的右边界，V指向相机的上方（视口的上边界），W从center指向target
         */
        Vec3 cameraU, cameraV, cameraW;

        Vec3 viewPortX, viewPortY;              //视口平面方向向量
        Vec3 viewPortPixelDx, viewPortPixelDy;  //视口平面方向变化向量，用于定位每一个像素的空间位置
        Point3 viewPortOrigin;                  //视口原点（左上角）空间坐标
        Point3 pixelOrigin;                     //第一个像素的空间坐标

        //采样属性
        double focusDiskRadius;                 //光线虚化强度，采样平面的圆盘半径
        double focusDistance;                   //焦距

        Range shutterRange;                     //相机快门的开启时间段

        double sampleRange;                     //采样偏移半径
        Uint32 sampleCount;                     //每像素采样数
        size_t sqrtSampleCount;
        double reciprocalSqrtSampleCount;

        Uint32 rayTraceDepth;                   //光线追踪深度

        __host__ Camera(Uint32 windowWidth, Uint32 windowHeight, const Color3 & backgroundColor,
               const Point3 & center, const Point3 & target, double fov, double focusDiskRadius,
               const Range & shutterRange, Uint32 sampleCount, double sampleRange,
               Uint32 rayTraceDepth, const Vec3 & upDirection);

        //重置相机位置
        __host__ void resetCameraPosition(const Point3 & newCenter, const Point3 & newTarget);

        //移动相机
        __host__ void shiftCameraPosition(const std::array<double, 3> & centerShift, const std::array<double, 3> & targetShift = {});

        __host__ std::string toString() const;
    };
}

#endif //RENDERERPARALLEL_CAMERA_CUH
