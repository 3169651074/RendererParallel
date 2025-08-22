#ifndef RENDERERPARALLEL_RANGE_CUH
#define RENDERERPARALLEL_RANGE_CUH

#include <Global.cuh>

namespace renderer {
    /*
     * 范围工具类，主机端和设备端通用
     */
    class Range {
    public:
        double min;
        double max;

        //默认构造空区间
        __host__ __device__ explicit Range(double min = 0.0, double max = 0.0) : min(min), max(max) {}

        //构造两个区间的并集
        __host__ __device__ Range(const Range & r1, const Range & r2) :
                min(r1.min < r2.min ? r1.min : r2.min), max(r1.max > r2.max ? r1.max : r2.max) {}

        __host__ __device__ Range(const Range & obj) {
            min = obj.min;
            max = obj.max;
        }

        __host__ __device__ Range & operator=(const Range & obj) {
            if (this == &obj) return *this;
            min = obj.min;
            max = obj.max;
            return *this;
        }

        __host__ __device__ ~Range() {}

        // ====== 对象操作函数 ======

        __host__ __device__ bool inRange(double value, bool isLeftClose = true, bool isRightClose = true) const {
            const bool equalsToMin = floatValueEquals(value, min);
            const bool equalsToMax = floatValueEquals(value, max);

            if ((value < min && !equalsToMin) || (value > max && !equalsToMax)) return false;
            if (equalsToMin && !isLeftClose) return false;
            if (equalsToMax && !isRightClose) return false;
            return true;
        }

        __host__ __device__ Range & offset(double offsetValue) {
            min += offsetValue;
            max += offsetValue;
            return *this;
        }

        __host__ __device__ bool isValid() const {
            return min < max || floatValueEquals(min, max);
        }

        __host__ __device__ double length() const {
            return max - min;
        }

        __host__ __device__ double clamp(double value) const {
            if (value > max) {
                return max;
            } else if (value < min) {
                return min;
            } else {
                return value;
            }
        }

        //将当前区间左右端点各扩展length长度
        __host__ __device__ Range & expand(double length) {
            if (length > 0) { //负长度不扩展
                min -= length;
                max += length;
            }
            return *this;
        }

        // ====== 类封装函数 =======

        __host__ std::string toString() const {
            char buffer[TOSTRING_BUFFER_SIZE] = { 0 };
            snprintf(buffer, TOSTRING_BUFFER_SIZE, "Range: [%.4lf, %.4lf]", min, max);
            return {buffer};
        }
    };
}

#endif //RENDERERPARALLEL_RANGE_CUH
