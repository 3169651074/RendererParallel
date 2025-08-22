#include <Camera.cuh>
#include <Render.cuh>
using namespace std;

namespace renderer {
    Camera::Camera(Uint32 windowWidth, Uint32 windowHeight, const Color3 & backgroundColor,
    const Point3 & center, const Point3 & target, double fov, double focusDiskRadius,
    const Range & shutterRange, Uint32 sampleCount, double sampleRange,
            Uint32 rayTraceDepth, const Vec3 & upDirection) :
            windowWidth(windowWidth), windowHeight(windowHeight), backgroundColor(backgroundColor),
            cameraCenter(center), cameraTarget(target), horizontalFOV(fov), focusDiskRadius(focusDiskRadius),
            shutterRange(shutterRange), sampleCount(sampleCount), sampleRange(sampleRange), upDirection(upDirection),
            rayTraceDepth(rayTraceDepth), focusDistance(Point3::distance(cameraCenter, cameraTarget))
    {
        const double thetaFOV = degreeToRadian(horizontalFOV);
        const double vWidth = 2.0 * tan(thetaFOV / 2.0) * focusDistance;
        const double vHeight = vWidth / (windowWidth * 1.0 / windowHeight);

        this->viewPortWidth = vWidth;
        this->viewPortHeight = vHeight;

        this->cameraW = Point3::constructVector(cameraCenter, cameraTarget).unitVector();
        this->cameraU = Vec3::cross(cameraW, upDirection).unitVector();
        this->cameraV = Vec3::cross(cameraU, cameraW).unitVector();

        this->viewPortX = vWidth * cameraU;
        this->viewPortY = vHeight * -cameraV;

        this->viewPortPixelDx = viewPortX / windowWidth;
        this->viewPortPixelDy = viewPortY / windowHeight;

        this->viewPortOrigin = cameraCenter + focusDistance * cameraW - viewPortX * 0.5 - viewPortY * 0.5;
        this->pixelOrigin = viewPortOrigin + viewPortPixelDx * 0.5 + viewPortPixelDy * 0.5;

        this->sqrtSampleCount = static_cast<size_t>(sqrt(sampleCount));
        this->reciprocalSqrtSampleCount = 1.0 / static_cast<double>(sqrtSampleCount);
    }

    __host__ void Camera::resetCameraPosition(const Point3 & newCenter, const Point3 & newTarget) {
        this->cameraCenter = newCenter;
        this->cameraTarget = newTarget;

        this->cameraW = Point3::constructVector(cameraCenter, cameraTarget).unitVector();
        this->cameraU = Vec3::cross(cameraW, upDirection).unitVector();
        this->cameraV = Vec3::cross(cameraU, cameraW).unitVector();

        const double thetaFOV = degreeToRadian(horizontalFOV);
        const double vWidth = 2.0 * tan(thetaFOV / 2.0) * focusDistance;
        const double vHeight = vWidth / (windowWidth * 1.0 / windowHeight);
        this->viewPortX = vWidth * cameraU;
        this->viewPortY = vHeight * -cameraV;

        this->viewPortPixelDx = viewPortX / windowWidth;
        this->viewPortPixelDy = viewPortY / windowHeight;

        this->viewPortOrigin = cameraCenter + focusDistance * cameraW - viewPortX * 0.5 - viewPortY * 0.5;
        this->pixelOrigin = viewPortOrigin + viewPortPixelDx * 0.5 + viewPortPixelDy * 0.5;
    }

    __host__ void Camera::shiftCameraPosition(const array<double, 3> & centerShift, const array<double, 3> & targetShift) {
        for (size_t i = 0; i < 3; i++) {
            this->cameraCenter[i] += centerShift[i];
            this->cameraTarget[i] += targetShift[i];
        }
        this->resetCameraPosition(cameraCenter, cameraTarget);
    }

    std::string Camera::toString() const {
        std::string ret("Renderer Camera:\n");
        char buffer[4 * TOSTRING_BUFFER_SIZE] = { 0 };
        snprintf(buffer, 4 * TOSTRING_BUFFER_SIZE,
                 "\tWindow Size: %u x %u\n\tBackground Color: %s\n\t"
                 "Camera Direction: %s --> %s, FOV: %.4lf\n\t"
                 "Viewport Size: %.4lf x %.4lf\n\t"
                 "Viewport Base Vector: U = %s, V = %s, W = %s\n\t"
                 "Viewport Delta Vector: dx = %s, dy = %s\n\t"
                 "Viewport Origin: %s, Pixel Origin: %s\n\t"
                 "Sample Disk Radius: %.4lf, Focus Distance: %.4lf\n\t"
                 "Shutter %s\n\tSSAA Sample Count: %u, Range: %.2lf\n\t"
                 "Raytrace Depth: %u",
                 windowWidth, windowHeight, backgroundColor.toString().c_str(),
                 cameraCenter.toString().c_str(), cameraTarget.toString().c_str(),
                 horizontalFOV, viewPortWidth, viewPortHeight,
                 cameraU.toString().c_str(), cameraV.toString().c_str(), cameraW.toString().c_str(),
                 viewPortPixelDx.toString().c_str(), viewPortPixelDy.toString().c_str(),
                 viewPortOrigin.toString().c_str(), pixelOrigin.toString().c_str(),
                 focusDiskRadius, focusDistance, shutterRange.toString().c_str(), sampleCount, sampleRange, rayTraceDepth
        );
        return ret + buffer;
    }
}