#include <Render.cuh>
using namespace std;

namespace renderer {
    void Renderer::setDirectSampleObject(const std::pair<PrimitiveType, size_t> * objectList, size_t objectListSize) {
        SDL_Log("Set direct sample object list.");
        if (objectList == nullptr) return;
        if (!this->devPointerAvailable) {
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

    void Renderer::renderInteractive(Camera * cam, SDL_Window * window) const {
        if (!devPointerAvailable) {
            SDL_LogError(SDL_LOG_CATEGORY_ERROR, "Device pointers not available!");
            return;
        }
        int w, h;
        SDL_GetWindowSize(window, &w, &h);
        SDL_DestroyWindow(window);

        window = SDL_CreateWindow("Test", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                  w, h, SDL_WINDOW_OPENGL);
        SDL_GLContext context = SDL_GL_CreateContext(window);

        if (!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress)) {
            SDL_Log("Failed to init glad");
            return;
        }
        glViewport(0, 0, w, h);
        GLuint textureID;
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);

        float vertices[] = {
                -1.0f,  1.0f,  0.0f, 1.0f, // Top-left
                -1.0f, -1.0f,  0.0f, 0.0f, // Bottom-left
                1.0f, -1.0f,  1.0f, 0.0f, // Bottom-right
                1.0f,  1.0f,  1.0f, 1.0f  // Top-right
        };
        Uint32 indices[] = {
                0, 1, 2,
                0, 2, 3
        };
        GLuint VAO, VBO, EBO;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);

        const char* vertexShaderSource = R"(
            #version 330 core
            layout (location = 0) in vec2 aPos;
            layout (location = 1) in vec2 aTexCoord;
            out vec2 TexCoord;
            void main() {
                gl_Position = vec4(aPos, 0.0, 1.0);
                TexCoord = aTexCoord;
            }
        )";

        const char* fragmentShaderSource = R"(
            #version 330 core
            out vec4 FragColor;
            in vec2 TexCoord;
            uniform sampler2D ourTexture;
            void main() {
                FragColor = texture(ourTexture, TexCoord);
            }
        )";
        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
        glCompileShader(vertexShader);
        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
        glCompileShader(fragmentShader);
        GLuint shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        cudaGraphicsResource_t cudaResource;
        cudaGraphicsGLRegisterImage(&cudaResource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

        Renderer * dev_renderer;
        cudaCheckError(cudaMalloc(&dev_renderer, sizeof(Renderer)));
        cudaCheckError(cudaMemcpy(dev_renderer, this, sizeof(Renderer), cudaMemcpyHostToDevice));
        Camera * dev_camera;
        cudaCheckError(cudaMalloc(&dev_camera, sizeof(Camera)));

        const dim3 blocks(cam->windowWidth % 16 == 0 ? cam->windowWidth / 16 : cam->windowWidth / 16 + 1,
                          cam->windowHeight % 16 == 0 ? cam->windowHeight / 16 : cam->windowHeight / 16 + 1, 1);
        const dim3 threads(16, 16, 1);
        const size_t pixelCount = w * h;
        curandState * dev_stateArray;
        cudaCheckError(cudaMalloc(&dev_stateArray, pixelCount * sizeof(curandState)));
        initThreadRandom<<<blocks, threads>>>(dev_stateArray);
        cudaCheckError(cudaDeviceSynchronize());

        bool quit = false;
        SDL_Event e;
        std::array<double, 3> centerShift = {};
        std::array<double, 3> targetShift = {};
        SDL_SetRelativeMouseMode(SDL_TRUE);

        while (!quit) {
            while (SDL_PollEvent(&e) != 0) {
                if (e.type == SDL_QUIT) {
                    quit = true;
                }
                
                if (e.type == SDL_KEYDOWN) {
                    const SDL_Keycode keycode = e.key.keysym.sym;
                    switch (keycode) {
                        case SDLK_a:
                            centerShift[0] = 0.1;
                            targetShift[0] = 0.1;
                            break;
                        case SDLK_d:
                            centerShift[0] = -0.1;
                            targetShift[0] = -0.1;
                            break;
                        case SDLK_w:
                            centerShift[2] = -0.1;
                            targetShift[2] = -0.1;
                            break;
                        case SDLK_s:
                            centerShift[2] = 0.1;
                            targetShift[2] = 0.1;
                            break;
                        case SDLK_SPACE:
                            centerShift[1] = 0.1;
                            targetShift[1] = 0.1;
                            break;
                        case SDLK_LSHIFT:
                            centerShift[1] = -0.1;
                            targetShift[1] = -0.1;
                            break;
                        default:;
                    }
                }
                if (e.type == SDL_KEYUP) {
                    centerShift[0] = targetShift[0] = 0.0;
                    centerShift[1] = targetShift[1] = 0.0;
                    centerShift[2] = targetShift[2] = 0.0;
                }
                if (e.type == SDL_MOUSEBUTTONDOWN) {
                    if (SDL_GetRelativeMouseMode() == SDL_TRUE) {
                        SDL_SetRelativeMouseMode(SDL_FALSE);
                        targetShift[1] = 0.0;
                    } else {
                        SDL_SetRelativeMouseMode(SDL_TRUE);
                    }
                }
                if (e.type == SDL_MOUSEMOTION && SDL_GetRelativeMouseMode() == SDL_TRUE) {
                    int dx = e.motion.xrel;
                    int dy = e.motion.yrel;
                    targetShift[1] = dy / -100.0;

                    Vec3 direction = Point3::constructVector(cam->cameraCenter, cam->cameraTarget);
                    double angle = (double)dx * -1e-3;
                    double currentDirX = direction[0];
                    double currentDirZ = direction[2];
                    direction[0] = currentDirX * cos(angle) - currentDirZ * sin(angle);
                    direction[2] = currentDirX * sin(angle) + currentDirZ * cos(angle);
                    cam->cameraTarget = cam->cameraCenter + direction;
                }
                cam->shiftCameraPosition(centerShift, targetShift);
                cudaCheckError(cudaMemcpy(dev_camera, cam, sizeof(Camera), cudaMemcpyHostToDevice));
            }

            // --- CUDA 计算阶段 ---
            // a. 映射资源，让CUDA接管纹理
            cudaGraphicsMapResources(1, &cudaResource, nullptr);

            // b. 获取指向纹理的CUDA数组
            cudaArray_t cudaTextureArray;
            cudaGraphicsSubResourceGetMappedArray(&cudaTextureArray, cudaResource, 0, 0);

            // c. 为CUDA数组创建一个 Surface Object，以便核函数写入
            cudaResourceDesc resDesc {};
            memset(&resDesc, 0, sizeof(resDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cudaTextureArray;
            cudaSurfaceObject_t surfaceObject;
            cudaCreateSurfaceObject(&surfaceObject, &resDesc);

            // d. 启动核函数
            renderToSurface<<<blocks, threads>>>(dev_renderer, dev_camera, surfaceObject, dev_stateArray);

            // e. 销毁 Surface Object
            cudaDestroySurfaceObject(surfaceObject);

            // f. 解除映射，将纹理控制权还给OpenGL
            cudaGraphicsUnmapResources(1, &cudaResource, nullptr);

            // --- OpenGL 渲染阶段 ---
            glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            glUseProgram(shaderProgram);
            glBindTexture(GL_TEXTURE_2D, textureID);
            glBindVertexArray(VAO);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

            SDL_GL_SwapWindow(window);
        }
        SDL_SetRelativeMouseMode(SDL_FALSE);

        cudaGraphicsUnregisterResource(cudaResource);

        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);
        glDeleteProgram(shaderProgram);
        glDeleteTextures(1, &textureID);

        SDL_GL_DeleteContext(context);

        cudaCheckError(cudaFree(dev_camera));
        cudaCheckError(cudaFree(dev_renderer));
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

            //SDL_Log("Clock rate: %d kHz", prop.clockRate);
            //SDL_Log("Memory clock rate: %d kHz", prop.memoryClockRate);

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