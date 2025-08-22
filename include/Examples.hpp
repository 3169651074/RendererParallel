#ifndef RENDERERPARALLEL_EXAMPLES_HPP
#define RENDERERPARALLEL_EXAMPLES_HPP

#include <Render.cuh>

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
}

namespace renderer {
    class Examples {
    public:
        //小球场景
        static void test01();

        //多种图元和材质
        static void test02();

        //康奈尔盒
        static void test03();
    };
}

#endif //RENDERERPARALLEL_EXAMPLES_HPP
