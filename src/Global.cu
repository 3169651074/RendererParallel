#include <Global.cuh>

using namespace renderer;

namespace renderer {
    //资源释放函数指针
    void (*releaseSDLResourcesFunc)() = nullptr;
}

namespace {
    void checkInitError(const char * taskName, TaskOnError task, bool isError) {
        if (releaseSDLResourcesFunc == nullptr) {
            SDL_LogError(SDL_LOG_CATEGORY_ERROR, "Release resources function is not set!");
        }

        if (isError) {
            switch (task) {
                case TaskOnError::PRINT_MESSAGE:
                    SDL_LogError(SDL_LOG_CATEGORY_ERROR, "%s failed! error message: %s", taskName, SDL_GetError());
                    break;
                case TaskOnError::EXIT_PROGRAM:
                    SDL_LogError(SDL_LOG_CATEGORY_ERROR, "%s failed! error message: %s, exit program!", taskName, SDL_GetError());
                    if (releaseSDLResourcesFunc != nullptr) {
                        releaseSDLResourcesFunc();
                    }
                    exit(EXIT_FAILURE);
                default:
                case TaskOnError::IGNORE:
                    break;
            }
        } else {
            SDL_Log("%s...OK", taskName);
        }
    }
}

namespace renderer {
    void SDL_CheckErrorInt(int retVal, const char * taskName, TaskOnError errorTask) {
        checkInitError(taskName, errorTask, retVal < 0);
    }

    void SDL_CheckErrorPtr(void * retVal, const char * taskName, TaskOnError errorTask) {
        checkInitError(taskName, errorTask, retVal == nullptr);
    }

    void SDL_registerReleaseResources(void (*f)()) {
        releaseSDLResourcesFunc = f;
    }
}