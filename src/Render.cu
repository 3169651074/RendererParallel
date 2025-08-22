#include <Render.cuh>
using namespace std;

namespace renderer {
    void Renderer::setDirectSampleObject(const std::pair<PrimitiveType, size_t> * objectList, size_t objectListSize) {
        SDL_Log("Set direct sample object list.");
        if (objectList == nullptr) return;
        if (dev_spheres == nullptr && dev_parallelograms == nullptr) {
            SDL_Log("Direct sample object list not initialized, call commitSceneData first.");
            return;
        }

        size_t sphereCount = 0, parallelogramCount = 0;
        //统计数量
        for (size_t i = 0; i < objectListSize; i++) {
            switch (objectList[i].first) {
                case PrimitiveType::SPHERE:
                    sphereCount++;
                    break;
                case PrimitiveType::PARALLELOGRAM:
                    parallelogramCount++;
                    break;
                default:;
            }
        }

        //分配显存
        cudaCheckError(cudaMalloc(&dev_hittablePDFSpheres, sphereCount * sizeof(Sphere *)));
        cudaCheckError(cudaMalloc(&dev_hittablePDFParallelograms, parallelogramCount * sizeof(Parallelogram *)));

        //写入数据，将已有物体的地址存入数组中
        vector<const Sphere *> sphereVector;
        vector<const Parallelogram *> parallelogramVector;

        for (size_t i = 0; i < objectListSize; i++) {
            switch (objectList[i].first) {
                case PrimitiveType::SPHERE:
                    sphereVector.push_back(&dev_spheres[objectList[i].second]);
                    break;
                case PrimitiveType::PARALLELOGRAM:
                    parallelogramVector.push_back(&dev_parallelograms[objectList[i].second]);
                    break;
                default:;
            }
        }

        //拷贝到显存
        cudaCheckError(cudaMemcpy(dev_hittablePDFSpheres, sphereVector.data(), sphereCount * sizeof(const Sphere *), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(dev_hittablePDFParallelograms, parallelogramVector.data(), parallelogramCount * sizeof(const Parallelogram *), cudaMemcpyHostToDevice));

        this->hittablePDFSphereCount = sphereCount;
        this->hittablePDFParallelogramCount = parallelogramCount;
        this->isDirectSample = true;
    }

    void Renderer::commitSceneData(const Rough * roughMaterials, Uint32 roughMaterialCount,
                         const Metal * metalMaterials, Uint32 metalMaterialCount,
                         const Dielectric * dielectricMaterials, Uint32 dielectricMaterialCount,
                         const DiffuseLight * diffuseLightMaterials, Uint32 diffuseLightMaterialCount,
                         const Sphere * spheres, Uint32 sphereCount,
                         const Triangle * triangles, Uint32 triangleCount,
                         const Parallelogram * parallelograms, Uint32 parallelogramCount,
                         const Box * boxs, Uint32 boxCount,
                         const Transform * transforms, Uint32 transformCount)
    {
        SDL_Log("Commit data...");
        SDL_Log("Constructing BVH...");

        //构建BVH，仅添加没有被变换的图元
#define _constructVector(className, arrayName) \
        vector<className> arrayName##Vector;\
        for (size_t i = 0; i < arrayName##Count; i++) {\
            if (!arrayName##s[i].isTransformed) arrayName##Vector.push_back(arrayName##s[i]);\
        }
        //============
        _constructVector(Sphere, sphere);
        _constructVector(Triangle, triangle);
        _constructVector(Parallelogram, parallelogram);
        _constructVector(Box, box);
        //============
#undef _constructVector
        const vector<Transform> transformVector(transforms, transforms + transformCount);

        //先利用vector的返回值传递接收数组，再转换为指针
        const auto ret = BVHTree::constructBVHTree(sphereVector, triangleVector, parallelogramVector, boxVector, transformVector);

        const auto tree = ret.first.data();
        const auto treeIndexArray = ret.second.data();

        const size_t treeSize = ret.first.size() * sizeof(BVHTree::BVHTreeNode);
        const size_t treeIndexArraySize = ret.second.size() * sizeof(pair<PrimitiveType, size_t>);

        //分配BVH树显存
        cudaCheckError(cudaMalloc(&dev_tree, treeSize));
        cudaCheckError(cudaMalloc(&dev_treeIndexArray, treeIndexArraySize));

        //拷贝BVH数据
        cudaCheckError(cudaMemcpy(dev_tree, tree, treeSize, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(dev_treeIndexArray, treeIndexArray, treeIndexArraySize, cudaMemcpyHostToDevice));

        //============

        //分配场景数据显存
        SDL_Log("Construction complete, allocating VRAM...");

        cudaCheckError(cudaMalloc(&dev_roughMaterials, roughMaterialCount * sizeof(Rough)));
        cudaCheckError(cudaMalloc(&dev_metalMaterials, metalMaterialCount * sizeof(Metal)));
        cudaCheckError(cudaMalloc(&dev_dielectricMaterials, dielectricMaterialCount * sizeof(Dielectric)));
        cudaCheckError(cudaMalloc(&dev_diffuseLightMaterials, diffuseLightMaterialCount * sizeof(DiffuseLight)));
        cudaCheckError(cudaMalloc(&dev_spheres, sphereCount * sizeof(Sphere)));
        cudaCheckError(cudaMalloc(&dev_triangles, triangleCount * sizeof(Triangle)));
        cudaCheckError(cudaMalloc(&dev_parallelograms, parallelogramCount * sizeof(Parallelogram)));
        cudaCheckError(cudaMalloc(&dev_boxes, boxCount * sizeof(Box)));
        cudaCheckError(cudaMalloc(&dev_transforms, transformCount * sizeof(Transform)));

        SDL_Log("VRAM allocation complete.");

        //拷贝场景数据
        SDL_Log("Copying data...");

        cudaCheckError(cudaMemcpy(dev_roughMaterials, roughMaterials, roughMaterialCount * sizeof(Rough), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(dev_metalMaterials, metalMaterials, metalMaterialCount * sizeof(Metal), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(dev_dielectricMaterials, dielectricMaterials, dielectricMaterialCount * sizeof(Dielectric), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(dev_diffuseLightMaterials, diffuseLightMaterials, diffuseLightMaterialCount * sizeof(DiffuseLight), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(dev_spheres, spheres, sphereCount * sizeof(Sphere), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(dev_triangles, triangles, triangleCount * sizeof(Triangle), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(dev_parallelograms, parallelograms, parallelogramCount * sizeof(Parallelogram), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(dev_boxes, boxs, boxCount * sizeof(Box), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(dev_transforms, transforms, transformCount * sizeof(Transform), cudaMemcpyHostToDevice));

        SDL_Log("Data copying complete.");

        //============

        this->devPointerAvailable = true;
    }

    void Renderer::freeSceneData() {
        //============

        //释放场景数据显存
        SDL_Log("Free VRAM...");

        cudaCheckError(cudaFree(dev_roughMaterials));
        cudaCheckError(cudaFree(dev_metalMaterials));
        cudaCheckError(cudaFree(dev_dielectricMaterials));
        cudaCheckError(cudaFree(dev_diffuseLightMaterials));
        cudaCheckError(cudaFree(dev_spheres));
        cudaCheckError(cudaFree(dev_triangles));
        cudaCheckError(cudaFree(dev_parallelograms));
        cudaCheckError(cudaFree(dev_boxes));
        cudaCheckError(cudaFree(dev_transforms));

        //============

        //释放BVH树显存
        cudaCheckError(cudaFree(dev_tree));
        cudaCheckError(cudaFree(dev_treeIndexArray));

        //释放采样物体指针数组显存
        cudaCheckError(cudaFree(dev_hittablePDFSpheres));
        cudaCheckError(cudaFree(dev_hittablePDFParallelograms));
        this->isDirectSample = false;

        SDL_Log("VRAM free success.");
        this->devPointerAvailable = false;
    }

    void Renderer::renderFrame(const Camera * cam, SDL_Window * window, bool isPrintInfo) const {
        //检查参数
        if (!devPointerAvailable) {
            SDL_LogError(SDL_LOG_CATEGORY_ERROR, "Device pointers not available!");
            return;
        }

        SDL_Surface * surface = SDL_GetWindowSurface(window);
        if (surface == nullptr) {
            SDL_LogError(SDL_LOG_CATEGORY_ERROR, "Surface pointer is nullptr!");
            return;
        }

        if (isPrintInfo) {
            SDL_Log("Pixel format: %s", SDL_GetPixelFormatName(surface->format->format));
        }

        //创建设备端像素缓冲区
        const size_t pixelCount = surface->w * surface->h;
        Uint32 * dev_pixelBuffer;
        cudaCheckError(cudaMalloc(&dev_pixelBuffer, pixelCount * sizeof(Uint32)));

        //拷贝设备端渲染器和相机对象
        Renderer * dev_renderer;
        Camera * dev_camera;
        cudaCheckError(cudaMalloc(&dev_renderer, sizeof(Renderer)));
        cudaCheckError(cudaMemcpy(dev_renderer, this, sizeof(Renderer), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMalloc(&dev_camera, sizeof(Camera)));
        cudaCheckError(cudaMemcpy(dev_camera, cam, sizeof(Camera), cudaMemcpyHostToDevice));

        //将整个屏幕划分为16x16的网格，每个网格对应一个block
        const dim3 blocks(cam->windowWidth % 16 == 0 ? cam->windowWidth / 16 : cam->windowWidth / 16 + 1,
                          cam->windowHeight % 16 == 0 ? cam->windowHeight / 16 : cam->windowHeight / 16 + 1, 1);
        const dim3 threads(16, 16, 1);

        //初始化线程随机数生成器
        curandState * dev_stateArray;
        cudaCheckError(cudaMalloc(&dev_stateArray, pixelCount * sizeof(curandState)));
        initThreadRandom<<<blocks, threads>>>(dev_stateArray);
        cudaCheckError(cudaDeviceSynchronize());

        //记录事件
        cudaEvent_t start, finish;
        cudaCheckError(cudaEventCreate(&start));
        cudaCheckError(cudaEventCreate(&finish));
        cudaCheckError(cudaEventRecord(start, nullptr));

        //启动渲染
        if (isPrintInfo) {
            SDL_Log("Rendering...");
        }
        render<<<blocks, threads>>>(dev_renderer, dev_camera, dev_pixelBuffer, dev_stateArray);
        //cudaCheckError(cudaDeviceSynchronize());

        //统计用时
        cudaCheckError(cudaEventRecord(finish, nullptr));
        cudaCheckError(cudaEventSynchronize(finish));
        float timeUsed;
        cudaCheckError(cudaEventElapsedTime(&timeUsed, start, finish));
        if (isPrintInfo) {
            SDL_Log("Render complete. Time: %.2fms", timeUsed);
        }
        cudaCheckError(cudaEventDestroy(start));
        cudaCheckError(cudaEventDestroy(finish));

        //拷贝缓冲区颜色到主机并显示，主机启动核函数后异步执行，必须等待核函数执行完毕
#define USING_BUFFER
#ifdef USING_BUFFER
        SDL_Delay(100);
        auto * pixelBuffer = new Uint32 [pixelCount];
        cudaCheckError(cudaMemcpy(pixelBuffer, dev_pixelBuffer, pixelCount * sizeof(Uint32), cudaMemcpyDeviceToHost));

        //将缓冲区中的颜色拷贝到surface
        memcpy(surface->pixels, pixelBuffer, pixelCount * sizeof(Uint32));
        SDL_UpdateWindowSurface(window);
        delete[] pixelBuffer;
#else
        //可以直接复制到surface
        cudaCheckError(cudaMemcpy(surface->pixels, dev_pixelBuffer, pixelCount * sizeof(Uint32), cudaMemcpyDeviceToHost));
        SDL_UpdateWindowSurface(window);
#endif

        //释放临时资源
        cudaCheckError(cudaFree(dev_pixelBuffer));
        cudaCheckError(cudaFree(dev_camera));
        cudaCheckError(cudaFree(dev_renderer));

        //保存渲染结果
        if (isPrintInfo) {
            SDL_CheckErrorInt(IMG_SavePNG(surface, "../files/output.png"), "Save PNG");
        }
    }

    void Renderer::printDeviceInfo() {
        SDL_Log("Querying devices...");

        cudaDeviceProp prop {};
        int deviceCount;

        cudaCheckError(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            SDL_LogError(SDL_LOG_CATEGORY_ERROR, "No CUDA device detected!");
            exit(EXIT_FAILURE);
        }

        SDL_Log("CUDA Version: %d.%d",  CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
        SDL_Log("==================================================");
        for (int i = 0; i < deviceCount; i++) {
            cudaCheckError(cudaGetDeviceProperties(&prop, i));
            SDL_Log("Device name: %s", prop.name);
            SDL_Log("Compute capability: %d.%d", prop.major, prop.minor);

            SDL_Log("Total global memory: %.2f MB", (double)prop.totalGlobalMem / (1024 * 1024));
            SDL_Log("Shared memory per block: %.2f KB", (double)prop.sharedMemPerBlock / 1024);
            SDL_Log("Reserved shared memory per block: %.2f KB", (double)prop.reservedSharedMemPerBlock / 1024);
            SDL_Log("Memory bus width: %d bits", prop.memoryBusWidth);
            SDL_Log("L2 cache size: %.2f KB", (double)prop.l2CacheSize / 1024);
            SDL_Log("Total constant memory: %.2f KB", (double)prop.totalConstMem / 1024);

            SDL_Log("Clock rate: %d kHz", prop.clockRate);
            SDL_Log("Memory clock rate: %d kHz", prop.memoryClockRate);

            SDL_Log("Registers per block: %d", prop.regsPerBlock);
            SDL_Log("Max threads per block: %d", prop.maxThreadsPerBlock);

            SDL_Log("Warp size: %d", prop.warpSize);
            SDL_Log("Multiprocessor count: %d", prop.multiProcessorCount);
            SDL_Log("Max blocks per multiprocessor: %d", prop.maxBlocksPerMultiProcessor);
            SDL_Log("Max threads per multiprocessor: %d", prop.maxThreadsPerMultiProcessor);
            SDL_Log("Shared memory per multiprocessor: %.2f KB", (double)prop.sharedMemPerMultiprocessor / 1024);
            SDL_Log("Registers per multiprocessor: %d", prop.regsPerMultiprocessor);

            SDL_Log("Max threads dimensions: (%d, %d, %d)",
                    prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
            SDL_Log("Max grid size: (%d, %d, %d)",
                    prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
            SDL_Log("Max texture dimensions: (%d, %d, %d)",
                    prop.maxTexture1D, prop.maxTexture2D[0], prop.maxTexture2D[1]);
            SDL_Log("Max surface dimensions: (%d, %d, %d)",
                    prop.maxSurface1D, prop.maxSurface2D[0], prop.maxSurface2D[1]);
            SDL_Log("==================================================");
        }
    }
}