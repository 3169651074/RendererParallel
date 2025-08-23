#include <Examples.hpp>
#include <Render.cuh>

using namespace renderer;
using namespace std;

namespace {

    //窗口比例
    constexpr double ASPECT_RATIO = 16.0 / 9.0;
    //窗口尺寸
    constexpr Uint32 WINDOW_WIDTH = 1200;
    constexpr Uint32 WINDOW_HEIGHT = static_cast<int>(WINDOW_WIDTH / ASPECT_RATIO);

    //SDL变量
    SDL_Window * window = nullptr;

    void initSDLResources();
    void releaseSDLResourcesImpl();

    void initSDLResources() {
        SDL_Log("Initializing renderer...");
        SDL_registerReleaseResources(releaseSDLResourcesImpl);
        int ret;

        ret = SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_EVENTS);
        SDL_CheckErrorInt(ret, "Init");
        window = SDL_CreateWindow("Test", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                  WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
        SDL_CheckErrorPtr(window, "Create Window");
        SDL_Log("SDL Version: %d.%d.%d", SDL_MAJOR_VERSION, SDL_MINOR_VERSION, SDL_PATCHLEVEL);
    }

    void releaseSDLResourcesImpl() {
        if (window != nullptr) {
            releaseSDLResource(SDL_DestroyWindow(window), "Destroy Window");
        }
        releaseSDLResource(SDL_Quit(), "Quit");
    }
}

namespace renderer {
    void Examples::test01() {
        initSDLResources();
        Renderer::printDeviceInfo();
        Renderer renderer;

        //============

        Camera cam(
                WINDOW_WIDTH, WINDOW_HEIGHT, Color3(0.7, 0.8, 1.0),
                Point3(0.0, 2.0, 10.0), Point3(0.0, 2.0, 0.0),
                80, 0.0, Range(0.0, 1.0), 10,
                0.5, 10, Vec3(0.0, 1.0, 0.0)
        );
        SDL_Log("%s", cam.toString().c_str());

        //动态分配材质数组
        vector<Rough> roughs;
        vector<Metal> metals;
        vector<Sphere> spheres;
        vector<Dielectric> dielectrics;

        //地面材质和球体
        roughs.emplace_back(Color3(0.7, 0.6, 0.5));
        spheres.emplace_back(MaterialType::ROUGH, 0, Point3(0.0, -1000.0, 0.0), 1000.0);

        int roughIndex = 1;
        int metalIndex = 0;

        //随机生成小球
        const int range = 4;
        for (int a = -range; a <= range; a++) {
            for (int b = -range; b <= range; b++) {
                double chooseMat = randomDoubleHost();
                Point3 center(a + 0.9 * randomDoubleHost(), 0.2, b + 0.9 * randomDoubleHost());

                if (Point3::constructVector(Point3(4.0, 0.2, 0.0), center).length() > 0.9) {
                    if (chooseMat < 0.8) {
                        //粗糙材质
                        auto albedo = Color3::randomColorHost() * Color3::randomColorHost();
                        roughs.emplace_back(albedo);

                        auto center2 = center + Vec3(0.0, randomDoubleHost(0.0, 0.5), 0.0);
                        spheres.emplace_back(MaterialType::ROUGH, roughIndex, center, center2, 0.2);
                        roughIndex++;
                    } else if (chooseMat < 0.95) {
                        //金属材质
                        auto albedo = Color3::randomColorHost(0.5, 1.0);
                        auto fuzz = randomDoubleHost(0.0, 0.5);
                        metals.emplace_back(albedo, fuzz);
                        spheres.emplace_back(MaterialType::METAL, metalIndex, center, 0.2);
                        metalIndex++;
                    } else {
                        dielectrics.emplace_back(1.5);
                        spheres.emplace_back(MaterialType::DIELECTRIC, 0, center, 0.2);
                        roughIndex++;
                    }
                }
            }
        }

        //添加三个大球体
        roughs.emplace_back(Color3(0.8, 0.9, 1.0));
        spheres.emplace_back(MaterialType::ROUGH, roughIndex, Point3(0.0, 1.0, 0.0), 1.0);
        roughIndex++;

        roughs.emplace_back(Color3(0.4, 0.2, 0.1));
        spheres.emplace_back(MaterialType::ROUGH, roughIndex, Point3(-4.0, 1.0, 0.0), 1.0);
        roughIndex++;

        metals.emplace_back(Color3(0.7, 0.6, 0.5), 0.0);
        spheres.emplace_back(MaterialType::METAL, metalIndex, Point3(4.0, 1.0, 0.0), 1.0);
        metalIndex++;

        //提交场景数据到渲染器
        renderer.commitSceneData(
                roughs.data(), roughs.size(),
                metals.data(), metals.size(),
                dielectrics.data(), dielectrics.size(),
                nullptr, 0,
                spheres.data(), spheres.size(),
                nullptr, 0,
                nullptr, 0,
                nullptr, 0,
                nullptr, 0
        );

        //============

        //启动渲染
        renderer.renderFrame(&cam, window);
        SDL_Delay(1000 * 2);

        //释放资源
        renderer.freeSceneData();
        releaseSDLResourcesImpl();
    }

    void Examples::test02() {
        initSDLResources();
        Renderer::printDeviceInfo();
        Renderer renderer;

        Camera cam(
                WINDOW_WIDTH, WINDOW_HEIGHT,
                Color3(0.7, 0.8, 0.9),
                Point3(0.0, 2.0, 10.0), Point3(0.0, 2.0, 0.0),
                80, 0.0, Range(0.0, 1.0), 100,
                0.5, 10, Vec3(0.0, 1.0, 0.0)
        );
        SDL_Log("%s", cam.toString().c_str());

        //材质列表
        const Rough roughs[] = {
                Rough(Color3(.65, .05, .05)),
                Rough(Color3(.73, .73, .73)),
                Rough(Color3(.12, .45, .15)),
                Rough(Color3(.70, .60, .50))
        };

        const Metal metals[] = {
                Metal(Color3(0.8, 0.85, 0.88), 0.0)
        };

        const Dielectric dielectrics[] = {
                Dielectric(1.5)
        };

        const DiffuseLight diffuseLights[] = {
                DiffuseLight(Color3(15.0, 15.0, 15.0))
        };

        //物体列表
        const Sphere spheres[] = {
                Sphere(MaterialType::ROUGH, 3, Point3(0.0, -1000.0, 0.0), 1000.0),
                Sphere(MaterialType::ROUGH, 0, Point3(0.0, 2.0, 0.0), 2.0)
        };

        const Parallelogram parallelograms[] = {
                Parallelogram(MaterialType::DIFFUSE_LIGHT, 0, Point3(-4.0, 0.0, 0.0), Vec3(0.5, 0.0, -1.0), Vec3(0.0, 4.0, 0.0))
        };

        const Triangle triangles[] = {
                Triangle(MaterialType::ROUGH, 2, Point3(4.0, 0.0, 0.0), Point3(5.5, 2.0, 0.0), Point3(4.0, 0.0, 3.0))
        };

#ifdef SAMPLE_LIGHT
        const std::pair<PrimitiveType, size_t> directSampleList[] = {
                {PrimitiveType::PARALLELOGRAM, 0}
        };
#endif

        renderer.commitSceneData(
                roughs, arrayLengthOnPos(roughs),
                metals, arrayLengthOnPos(metals),
                dielectrics, arrayLengthOnPos(dielectrics),
                diffuseLights, arrayLengthOnPos(diffuseLights),
                spheres, arrayLengthOnPos(spheres),
                triangles, arrayLengthOnPos(triangles),
                parallelograms, arrayLengthOnPos(parallelograms),
                nullptr, 0,
                nullptr, 0
        );

#ifdef SAMPLE_LIGHT
        renderer.setDirectSampleObject(directSampleList, arrayLengthOnPos(directSampleList));
#endif

        renderer.renderFrame(&cam, window);
        SDL_Delay(1000 * 2);

//        SDL_Event event{};
//        bool isQuit = false;
//
//        std::array<double, 3> centerShift = {};
//        std::array<double, 3> targetShift = {};
//
//        SDL_SetRelativeMouseMode(SDL_TRUE); 开启相对鼠标模式（锁定+隐藏光标）
//        while (!isQuit) {
//            while (SDL_PollEvent(&event)) {
//                if (event.type == SDL_QUIT) isQuit = true;
//
//                if (event.type == SDL_KEYDOWN) {
//                    const SDL_Keycode keycode = event.key.keysym.sym;
//                    switch (keycode) {
//                        case SDLK_a:
//                            centerShift[0] = -0.1;
//                            targetShift[0] = -0.1;
//                            break;
//                        case SDLK_d:
//                            centerShift[0] = 0.1;
//                            targetShift[0] = 0.1;
//                            break;
//                        case SDLK_w:
//                            centerShift[2] = -0.1;
//                            targetShift[2] = -0.1;
//                            break;
//                        case SDLK_s:
//                            centerShift[2] = 0.1;
//                            targetShift[2] = 0.1;
//                            break;
//                        case SDLK_SPACE:
//                            centerShift[1] = 0.1;
//                            targetShift[1] = 0.1;
//                            break;
//                        case SDLK_LSHIFT:
//                            centerShift[1] = -0.1;
//                            targetShift[1] = -0.1;
//                            break;
//                        default:;
//                    }
//                }
//
//                if (event.type == SDL_KEYUP) {
//                    centerShift[0] = targetShift[0] = 0.0;
//                    centerShift[1] = targetShift[1] = 0.0;
//                    centerShift[2] = targetShift[2] = 0.0;
//                }
//
//                if (event.type == SDL_MOUSEBUTTONDOWN) {
//                    if (SDL_GetRelativeMouseMode() == SDL_TRUE) {
//                        SDL_SetRelativeMouseMode(SDL_FALSE);
//                        targetShift[1] = 0.0;
//                    } else {
//                        SDL_SetRelativeMouseMode(SDL_TRUE);
//                    }
//                }
//
//                if (event.type == SDL_MOUSEMOTION && SDL_GetRelativeMouseMode() == SDL_TRUE) {
//                    int dx = event.motion.xrel; // 水平方向位移
//                    int dy = event.motion.yrel; // 竖直方向位移
//                    targetShift[1] = dy / -100.0;
//                }
//                cam.shiftCameraPosition(centerShift, targetShift);
//            }
//            renderer.renderFrame(&cam, window, false);
//        }

        renderer.freeSceneData();
        releaseSDLResourcesImpl();
    }

    void Examples::test03() {
        //SDL_Log和clog都默认输出到stderr，clog缓冲，cerr不缓冲
        initSDLResources();
        Renderer::printDeviceInfo();
        Renderer renderer;

        Camera cam(
                WINDOW_WIDTH, WINDOW_HEIGHT,
                Color3(),
                Point3(278, 278, -600), Point3(278, 278, 0),
                80, 0.0, Range(0.0, 1.0), 10,
                0.5, 50, Vec3(0.0, 1.0, 0.0)
        );

        //材质列表
        const Rough roughs[] = {
                Rough(Color3(.65, .05, .05)),
                Rough(Color3(.73, .73, .73)),
                Rough(Color3(.12, .45, .15))
        };

        const Metal metals[] = {
                Metal(Color3(0.8, 0.85, 0.88), 0.0)
        };

        const DiffuseLight diffuseLights[] = {
                DiffuseLight(Color3(25.0, 25.0, 25.0))
        };

        const Dielectric dielectrics[] = {
                Dielectric(1.5)
        };

        //物体列表
        const Sphere spheres[] = {
                Sphere(MaterialType::DIELECTRIC, 0, Point3(190.0, 90.0, 190.0), 90.0)
        };

        //盒子，启用被变换标记
        const Box boxes[] = {
                Box(MaterialType::ROUGH, 1, Point3(), Point3(165.0, 165.0, 165.0), true),
                Box(MaterialType::METAL, 0, Point3(), Point3(165.0, 330.0, 165.0), true)
        };

        //墙壁和灯光
        const Parallelogram parallelograms[] = {
                Parallelogram(MaterialType::ROUGH, 2, Point3(555.0, 0.0, 0.0), Vec3(0.0, 0.0, 555.0), Vec3(0.0, 555.0, 0.0)),
                Parallelogram(MaterialType::ROUGH, 0, Point3(0.0, 0.0, 555.0), Vec3(0.0, 0.0, -555.0), Vec3(0.0, 555.0, 0.0)),
                Parallelogram(MaterialType::ROUGH, 1, Point3(0.0, 555.0, 0.0), Vec3(555.0, 0.0, 0.0), Vec3(0.0, 0.0, 555.0)),
                Parallelogram(MaterialType::ROUGH, 1, Point3(0.0, 0.0, 555.0), Vec3(555.0, 0.0, 0.0), Vec3(0.0, 0.0, -555.0)),
                Parallelogram(MaterialType::ROUGH, 1, Point3(555.0, 0.0, 555.0), Vec3(-555.0, 0.0, 0.0), Vec3(0.0, 555.0, 0.0)),
                Parallelogram(MaterialType::DIFFUSE_LIGHT, 0, Point3(213.0, 554.0, 227.0), Vec3(130.0, 0.0, 0.0), Vec3(0.0, 0.0, 105.0))
        };

        //变换
        const Transform transforms[] = {
                //Transform(PrimitiveType::BOX, 0, boxes[0].constructBoundingBox(), boxes[0].centroid(), std::array<double, 3>{0.0, -15.0, 0.0}, std::array<double, 3>{130, 0.0, 65.0}/*, std::array<double, 3>{1.5, 1.5, 1.5}*/),
                Transform(PrimitiveType::BOX, 1, boxes[1].constructBoundingBox(), boxes[1].centroid(), std::array<double, 3>{0.0, 18.0, 0.0}, std::array<double, 3>{265.0, 0.0, 295.0}/*, std::array<double, 3>{1.5, 1.5, 1.5}*/)
        };

        //直接重要性采样物体列表，引用已存在的图元对象，无需重新创建
        const std::pair<PrimitiveType, size_t> directSampleList[] = {
                {PrimitiveType::PARALLELOGRAM, 5}
                ,{PrimitiveType::SPHERE, 0}
        };

        renderer.commitSceneData(
                roughs, arrayLengthOnPos(roughs),
                metals, arrayLengthOnPos(metals),
                dielectrics, arrayLengthOnPos(dielectrics),
                diffuseLights, arrayLengthOnPos(diffuseLights),
                spheres, arrayLengthOnPos(spheres),
                nullptr, 0,
                parallelograms, arrayLengthOnPos(parallelograms),
                boxes, arrayLengthOnPos(boxes),
                transforms, arrayLengthOnPos(transforms)
        );
        renderer.setDirectSampleObject(directSampleList, arrayLengthOnPos(directSampleList));

        renderer.renderFrame(&cam, window);
        SDL_Delay(1000 * 2);
        renderer.freeSceneData();
        releaseSDLResourcesImpl();
    }
}