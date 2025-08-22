#ifndef RENDERERPARALLEL_VEC3_CUH
#define RENDERERPARALLEL_VEC3_CUH

#include <Global.cuh>

namespace renderer {
    /*
     * 三维向量类，主机端和设备端通用
     */
    class Vec3 {
    private:
        double elements[3] {};

    public:
        static constexpr double VECTOR_LENGTH_SQUARE_ZERO_EPSILON = FLOAT_VALUE_ZERO_EPSILON * FLOAT_VALUE_ZERO_EPSILON;

        //显式提供构造和拷贝构造，以及赋值运算符和析构
        __host__ __device__ explicit Vec3(double x = 0.0, double y = 0.0, double z = 0.0) {
            elements[0] = x; elements[1] = y; elements[2] = z;
        }

        __host__ __device__ Vec3(const Vec3 & obj) {
            for (size_t i = 0; i < 3; i++) {
                elements[i] = obj.elements[i];
            }
        }

        __host__ __device__ Vec3 & operator=(const Vec3 & obj) {
            if (this == &obj) return *this;
            for (size_t i = 0; i < 3; i++) {
                elements[i] = obj.elements[i];
            }
            return *this;
        }

        __host__ __device__ ~Vec3() {}

        //成员访问函数

        __host__ __device__ double & operator[](size_t index) {
            return elements[index];
        }
        __host__ __device__ double operator[](size_t index) const {
            return elements[index];
        }

        // ====== 对象操作函数 ======

        //复制当前向量的负向量
        __host__ __device__ Vec3 operator-() const {
            return Vec3(-elements[0], -elements[1], -elements[2]);
        }

        //将当前向量取负，修改当前向量
        __host__ __device__ Vec3 & negate() {
            for (double & element : elements) {
                element = -element;
            }
            return *this;
        }

        //向量加减
        __host__ __device__ Vec3 & operator+=(const Vec3 & obj) {
            for (size_t i = 0; i < 3; i++) {
                elements[i] += obj.elements[i];
            }
            return *this;
        }

        __host__ __device__ Vec3 operator+(const Vec3 & obj) const {
            Vec3 ret(*this); ret += obj; return ret;
        }

        __host__ __device__ Vec3 & operator-=(const Vec3 & obj) {
            for (size_t i = 0; i < 3; i++) {
                elements[i] -= obj.elements[i];
            }
            return *this;
        }

        __host__ __device__ Vec3 operator-(const Vec3 & obj) const {
            Vec3 ret(*this); ret -= obj; return ret;
        }

        //向量数乘除
        __host__ __device__ Vec3 & operator*=(double num) {
            for (double & element : elements) {
                element *= num;
            }
            return *this;
        }

        __host__ __device__ Vec3 operator*(double num) const {
            Vec3 ret(*this); ret *= num; return ret;
        }

        __host__ __device__ Vec3 & operator/=(double num) {
            for (double & element : elements) {
                element /= num;
            }
            return *this;
        }

        __host__ __device__ Vec3 operator/(double num) const {
            Vec3 ret(*this); ret /= num; return ret;
        }

        //数乘除操作允许左操作数为实数
        __host__ __device__ friend Vec3 operator*(double num, const Vec3 & obj) {
            Vec3 ret(obj); ret *= num; return ret;
        }

        __host__ __device__ friend Vec3 operator/(double num, const Vec3 & obj) {
            Vec3 ret(obj); ret /= num; return ret;
        }

        __host__ __device__ double lengthSquare() const {
            double sum = 0.0;
            for (const double & element : elements) {
                sum += element * element;
            }
            return sum;
        }

        __host__ __device__ double length() const {
            return sqrt(lengthSquare());
        }

        __host__ __device__ double dot(const Vec3 & obj) const {
            double sum = 0.0;
            for (size_t i = 0; i < 3; i++) {
                sum += elements[i] * obj.elements[i];
            }
            return sum;
        }

        __host__ __device__ Vec3 cross(const Vec3 & obj) const {
            return Vec3(elements[1] * obj.elements[2] - elements[2] * obj.elements[1],
                        elements[2] * obj.elements[0] - elements[0] * obj.elements[2],
                        elements[0] * obj.elements[1] - elements[1] * obj.elements[0]);
        }

        __host__ __device__ Vec3 & unitize() {
            const double len = length();
            for (double & element : elements) {
                element /= len;
            }
            return *this;
        }

        __host__ __device__ Vec3 unitVector() const {
            Vec3 ret(*this); ret.unitize(); return ret;
        }

        // ====== 静态操作函数 ======

        //生成遵守按指定轴余弦分布的随机向量，非单位向量，主机版本
        __host__ static inline Vec3 randomCosineVectorHost(int axis, bool toPositive) {
            double coord[3];
            const auto r1 = randomDoubleHost();
            const auto r2 = randomDoubleHost();

            coord[0] = cos(2.0 * PI * r1) * 2.0 * sqrt(r2);
            coord[1] = sin(2.0 * PI * r1) * 2.0 * sqrt(r2);
            coord[2] = sqrt(1.0 - r2);

            switch (axis) {
                case 0:
                    std::swap(coord[0], coord[2]);
                    break;
                case 1:
                    std::swap(coord[1], coord[2]);
                    break;
                case 2:
                default:
                    break;
            }

            if (!toPositive) {
                coord[axis] = -coord[axis];
            }
            return Vec3(coord[0], coord[1], coord[2]);
        }

        //生成遵守按指定轴余弦分布的随机向量，非单位向量，主机版本
        __device__ static inline Vec3 randomCosineVectorDevice(curandState * state, int axis, bool toPositive) {
            double coord[3];
            const auto r1 = randomDoubleDevice(state);
            const auto r2 =  randomDoubleDevice(state);

            coord[0] = cos(2.0 * PI * r1) * 2.0 * sqrt(r2);
            coord[1] = sin(2.0 * PI * r1) * 2.0 * sqrt(r2);
            coord[2] = sqrt(1.0 - r2);

            switch (axis) {
                case 0: {
                    const double tmp = coord[0];
                    coord[0] = coord[2];
                    coord[2] = tmp;
                    break;
                }
                case 1: {
                    const double tmp = coord[1];
                    coord[1] = coord[2];
                    coord[2] = tmp;
                    break;
                }
                case 2:
                default:
                    break;
            }

            if (!toPositive) {
                coord[axis] = -coord[axis];
            }
            return Vec3(coord[0], coord[1], coord[2]);
        }

        //生成每个分量都在指定范围内的随机向量，主机版本
        __host__ static inline Vec3 randomVectorHost(double componentMin, double componentMax) {
            Vec3 ret;
            for (size_t i = 0; i < 3; i++) {
                ret[i] = randomDoubleHost(componentMin, componentMax);
            }
            return ret;
        }

        //生成每个分量都在指定范围内的随机向量，设备版本
        __device__ static inline Vec3 randomVectorDevice(curandState * state, double componentMin, double componentMax) {
            Vec3 ret;
            for (size_t i = 0; i < 3; i++) {
                ret[i] = randomDoubleDevice(state, componentMin, componentMax);
            }
            return ret;
        }

        //生成平面（x，y，0）上模长不大于maxLength的向量，主机版本
        __host__ static inline Vec3 randomPlaneVectorHost(double maxLength) {
            double x, y;
            do {
                x = randomDoubleHost(-1.0, 1.0);
                y = randomDoubleHost(-1.0, 1.0);
            } while (x * x + y * y > maxLength * maxLength);
            return Vec3(x, y, 0.0);
        }

        //生成平面（x，y，0）上模长不大于maxLength的向量，设备版本
        __device__ static inline Vec3 randomPlaneVectorDevice(curandState * state, double maxLength) {
            double x, y;
            do {
                x = randomDoubleDevice(state, -1.0, 1.0);
                y = randomDoubleDevice(state, -1.0, 1.0);
            } while (x * x + y * y > maxLength * maxLength);
            return Vec3(x, y, 0.0);
        }

        //生成模长为length的空间向量，主机版本
        __host__ static inline Vec3 randomSpaceVectorHost(double length) {
            Vec3 ret;
            double lengthSquare;
            //先生成单位向量，再缩放到指定模长
            do {
                for (size_t i = 0; i < 3; i++) {
                    ret[i] = randomDoubleHost(-1.0, 1.0);
                }
                lengthSquare = ret.lengthSquare();
            } while (lengthSquare < VECTOR_LENGTH_SQUARE_ZERO_EPSILON);

            //单位化ret，确定模长
            ret.unitize();
            return ret * length;
        }

        //生成模长为length的空间向量，主机版本
        __device__ static inline Vec3 randomSpaceVectorDevice(curandState * state, double length) {
            Vec3 ret;
            double lengthSquare;
            do {
                for (size_t i = 0; i < 3; i++) {
                    ret[i] = randomDoubleDevice(state, -1.0, 1.0);
                }
                lengthSquare = ret.lengthSquare();
            } while (lengthSquare < VECTOR_LENGTH_SQUARE_ZERO_EPSILON);
            ret.unitize();
            return ret * length;
        }

        // ====== 通用静态操作函数 ======

        __host__ __device__ static inline Vec3 negativeVector(const Vec3 & obj) {
            return -obj;
        }

        __host__ __device__ static inline Vec3 add(const Vec3 & v1, const Vec3 & v2) {
            return v1 + v2;
        }

        __host__ __device__ static inline Vec3 subtract(const Vec3 & origin, const Vec3 & sub) {
            return origin - sub;
        }

        __host__ __device__ static inline Vec3 multiply(const Vec3 & obj, double num) {
            return obj * num;
        }

        __host__ __device__ static inline Vec3 divide(const Vec3 & origin, double num) {
            return origin / num;
        }

        __host__ __device__ static inline double lengthSquare(const Vec3 & obj) {
            return obj.lengthSquare();
        }

        __host__ __device__ static inline double length(const Vec3 & obj) {
            return obj.length();
        }

        __host__ __device__ static inline double dot(const Vec3 & v1, const Vec3 & v2) {
            return v1.dot(v2);
        }

        //v1 x v2
        __host__ __device__ static inline Vec3 cross(const Vec3 & v1, const Vec3 & v2) {
            return v1.cross(v2);
        }

        __host__ __device__ static inline Vec3 unitVector(const Vec3 & obj) {
            return obj.unitVector();
        }

        // ====== 类封装函数 ======

        //设备端不支持std::string，此函数只能被主机端调用
        __host__ std::string toString() const {
            char buffer[TOSTRING_BUFFER_SIZE] = { 0 };
            snprintf(buffer, TOSTRING_BUFFER_SIZE, "Vec3: (%.4lf, %.4lf, %.4lf)", elements[0], elements[1], elements[2]);
            return {buffer};
        }
    };
}

#endif //RENDERERPARALLEL_VEC3_CUH
