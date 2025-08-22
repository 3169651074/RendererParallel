#ifndef RENDERERPARALLEL_COLOR3_CUH
#define RENDERERPARALLEL_COLOR3_CUH

#include <util/Range.cuh>

namespace renderer {
    /*
     * 颜色类，主机端和设备端通用
     */
    class Color3 {
    private:
        double elements[3] {};

    public:
        //主机端和设备端均允许构造颜色对象
        __host__ __device__ explicit Color3(double r = 0.0, double g = 0.0, double b = 0.0) {
            elements[0] = r; elements[1] = g; elements[2] = b;
        }

        __host__ __device__ Color3(const Color3 & obj) {
            for (size_t i = 0; i < 3; i++) {
                elements[i] = obj.elements[i];
            }
        }

        __host__ __device__ Color3 & operator=(const Color3 & obj) {
            if (this == &obj) return *this;
            for (size_t i = 0; i < 3; i++) {
                elements[i] = obj.elements[i];
            }
            return *this;
        }

        __host__ __device__ ~Color3() {}

        //成员访问函数

        __host__ __device__ double & operator[](size_t index) {
            return elements[index];
        }
        __host__ __device__ double operator[](size_t index) const {
            return elements[index];
        }

        // ====== 对象操作函数 ======

        //颜色相加减
        __host__ __device__ Color3 & operator+=(const Color3 & obj) {
            for (size_t i = 0; i < 3; i++) {
                elements[i] += obj.elements[i];
            }
            return *this;
        }

        __host__ __device__ Color3 operator+(const Color3 & obj) const {
            Color3 ret(*this); ret += obj; return ret;
        }

        __host__ __device__ Color3 & operator-=(const Color3 & obj) {
            for (size_t i = 0; i < 3; i++) {
                elements[i] -= obj.elements[i];
            }
            return *this;
        }

        __host__ __device__ Color3 operator-(const Color3 & obj) const {
            Color3 ret(*this); ret -= obj; return ret;
        }

        //颜色数乘除
        __host__ __device__ Color3 & operator*=(double num) {
            for (double & element : elements) {
                element *= num;
            }
            return *this;
        }

        __host__ __device__ Color3 operator*(double num) const {
            Color3 ret(*this); ret *= num; return ret;
        }

        __host__ __device__ Color3 & operator/=(double num) {
            for (double & element : elements) {
                element /= num;
            }
            return *this;
        }

        __host__ __device__ Color3 operator/(double num) const {
            Color3 ret(*this); ret /= num; return ret;
        }

        //数乘除允许左操作数为实数
        __host__ __device__ friend Color3 operator*(double num, const Color3 & obj) {
            return obj * num;
        }

        __host__ __device__ friend Color3 operator/(double num, const Color3 & obj) {
            return obj / num;
        }

        //颜色相乘除
        __host__ __device__ Color3 & operator*=(const Color3 & obj) {
            for (size_t i = 0; i < 3; i++) {
                elements[i] *= obj.elements[i];
            }
            return *this;
        }

        __host__ __device__ Color3 operator*(const Color3 & obj) const {
            Color3 ret(*this); ret *= obj; return ret;
        }

        __host__ __device__ Color3 & operator/=(const Color3 & obj) {
            for (size_t i = 0; i < 3; i++) {
                elements[i] /= obj.elements[i];
            }
            return *this;
        }

        __host__ __device__ Color3 operator/(const Color3 & obj) const {
            Color3 ret(*this); ret /= obj; return ret;
        }

        //颜色写入函数，设备函数
        __device__ void writeColor(Uint8 * pixelPointer, double gamma = 2.0) const {
            //伽马校正
            const double power = 1.0 / gamma;
            const double r = std::pow(elements[0], power);
            const double g = std::pow(elements[1], power);
            const double b = std::pow(elements[2], power);

            //将[0.0, 1.0]的颜色值映射到[0, 255]并写入
            const Range intensity(0.0, 0.999);
            const auto r_byte = static_cast<Uint8>(256 * intensity.clamp(r));
            const auto g_byte = static_cast<Uint8>(256 * intensity.clamp(g));
            const auto b_byte = static_cast<Uint8>(256 * intensity.clamp(b));

            //SDL_PIXELFORMAT_RGB888
            pixelPointer[2] = r_byte;
            pixelPointer[1] = g_byte;
            pixelPointer[0] = b_byte;
        }

        // ====== 静态操作函数 ======

        //生成随机颜色，主机函数
        __host__ static Color3 randomColorHost(double min = 0.0, double max = 1.0) {
            Color3 ret;
            for (size_t i = 0; i < 3; i++) {
                ret[i] = randomDoubleHost(min, max);
            }
            return ret;
        }

        //生成随机颜色，设备函数
        __device__ static Color3 randomColorDevice(curandState  * state, double min = 0.0, double max = 1.0) {
            Color3 ret;
            for (size_t i = 0; i < 3; i++) {
                ret[i] = randomDoubleDevice(state, min, max);
            }
            return ret;
        }

        // ====== 类封装函数 ======

        __host__ std::string toString() const {
            char buffer[TOSTRING_BUFFER_SIZE] = { 0 };
            snprintf(buffer, TOSTRING_BUFFER_SIZE, "Color3: (%.2lf, %.2lf, %.2lf)", elements[0], elements[1], elements[2]);
            return {buffer};
        }
    };
}

#endif //RENDERERPARALLEL_COLOR3_CUH
