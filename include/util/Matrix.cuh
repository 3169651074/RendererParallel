#ifndef RENDERERPARALLEL_MATRIX_CUH
#define RENDERERPARALLEL_MATRIX_CUH

#include <basic/Point3.cuh>

namespace renderer {
    /*
     * POD矩阵类，包含基础的矩阵运算
     * 由CPU（构造Transform类时计算逆矩阵和转置）和GPU线程串行执行（hit方法内部的矩阵乘法）
     */
    class Matrix {
    public:
        //使用固定大小的数组代替指针，限定矩阵最大大小为4x4，1基
        double data[5][5] {};
        size_t row, col;

        // ====== 构造析构函数 ======

        //构造函数，拷贝构造，赋值运算符和析构函数均为两端通用
        __host__ __device__ explicit Matrix(size_t rowCount = 4, size_t colCount = 4) : row(rowCount), col(colCount) {}

        //使用初始化列表创建矩阵，仅用于主机端测试
        __host__ Matrix(size_t rowCount, size_t colCount, std::initializer_list<double> list) : row(rowCount), col(colCount) {
            std::initializer_list<double>::const_iterator it = list.begin();
            for (size_t i = 1; i < 5; i++) {
                for (size_t j = 1; j < 5; j++) {
                    if (it != list.end()) {
                        data[i][j] = *it;
                        it++;
                    } //给出的参数数量不够时，自动使用类型默认值
                }
            }
        }

        __host__ __device__ Matrix(const Matrix & obj) : row(obj.row), col(obj.col) {
            //兼容两端，不使用内存拷贝函数
            for (size_t i = 1; i < 5; i++) {
                for (size_t j = 1; j < 5; j++) {
                    data[i][j] = obj.data[i][j];
                }
            }
        }

        __host__ __device__ Matrix & operator=(const Matrix & obj) {
            if (this == &obj) return *this;
            row = obj.row; col = obj.col;

            for (size_t i = 1; i < 5; i++) {
                for (size_t j = 1; j < 5; j++) {
                    data[i][j] = obj.data[i][j];
                }
            }
            return *this;
        }

        __host__ __device__ ~Matrix() {}

        // ====== 矩阵运算函数 ======

        //转置当前矩阵，返回新矩阵
        __host__ __device__ Matrix transpose() const;

        //矩阵乘法
        __host__ __device__ Matrix operator*(const Matrix & right) const;

        //求逆矩阵
        __host__ __device__ Matrix inverse() const;

        // ====== 矩阵简单运算函数 ======

        //矩阵加减
        __host__ __device__ Matrix & operator+=(const Matrix & obj) {
            for (size_t i = 1; i < 5; i++) {
                for (size_t j = 1; j < 5; j++) {
                    data[i][j] += obj.data[i][j];
                }
            }
            return *this;
        }
        __host__ __device__ Matrix operator+(const Matrix & obj) const {
            Matrix ret(*this); ret += obj; return ret;
        }

        __host__ __device__ Matrix & operator-=(const Matrix & obj) {
            for (size_t i = 1; i < 5; i++) {
                for (size_t j = 1; j < 5; j++) {
                    data[i][j] -= obj.data[i][j];
                }
            }
            return *this;
        }
        __host__ __device__ Matrix operator-(const Matrix & obj) const {
            Matrix ret(*this); ret -= obj; return ret;
        }

        //矩阵数乘除，允许左操作数为数字
        __host__ __device__ Matrix & operator*=(double num) {
            for (size_t i = 1; i < 5; i++) {
                for (size_t j = 1; j < 5; j++) {
                    data[i][j] *= num;
                }
            }
            return *this;
        }
        __host__ __device__ Matrix operator*(double num) const {
            Matrix ret(*this); ret *= num; return ret;
        }
        __host__ __device__ friend Matrix operator*(double num, const Matrix & obj) {
            return obj * num;
        }

        __host__ __device__ Matrix & operator/=(double num) {
            for (size_t i = 1; i < 5; i++) {
                for (size_t j = 1; j < 5; j++) {
                    data[i][j] /= num;
                }
            }
            return *this;
        }
        __host__ __device__ Matrix operator/(double num) const {
            Matrix ret(*this); ret /= num; return ret;
        }
        __host__ __device__ friend Matrix operator/(double num, const Matrix & obj) {
            return obj / num;
        }

        // ====== 辅助函数 ======

        //用于使用初等变换法球合并后的矩阵的逆矩阵，修改参数数组
        __host__ __device__ static int eliminateBottomElements(double matrixData[5][9]);
        __host__ __device__ static int eliminateTopElements(double matrixData[5][9]);

        // ====== 静态操作函数 ======

        //使用空间点或空间向量构造4行1列矩阵
        __host__ __device__ static inline Matrix toMatrix(const Vec3 & obj) {
            Matrix ret(4, 1);
            for (size_t i = 0; i < 3; i++) {
                ret.data[i + 1][1] = obj[i];
            }
            return ret;
        }
        __host__ __device__ static inline Matrix toMatrix(const Point3 & obj) {
            Matrix ret(4, 1);
            for (size_t i = 0; i < 3; i++) {
                ret.data[i + 1][1] = obj[i];
            }
            ret.data[4][1] = 1.0; //点有位置属性，向量没有
            return ret;
        }

        //将4行1列矩阵转为空间点或空间向量
        __host__ __device__ static inline Point3 toPoint(const Matrix & obj) {
            return Point3(obj.data[1][1], obj.data[2][1], obj.data[3][1]);
        }
        __host__ __device__ static inline Vec3 toVector(const Matrix & obj) {
            return Vec3(obj.data[1][1], obj.data[2][1], obj.data[3][1]);
        }

        //构造三维平移矩阵
        __host__ static Matrix constructShiftMatrix(const std::array<double, 3> & shift);

        //构造三维缩放矩阵
        __host__ static Matrix constructScaleMatrix(const std::array<double, 3> & scale);

        //构造三维旋转矩阵。0，1，2表示x，y，z轴
        __host__ static Matrix constructRotateMatrix(double degree, int axis);
        __host__ static Matrix constructRotateMatrix(const std::array<double, 3> & rotate);
    };
}

#endif //RENDERERPARALLEL_MATRIX_CUH
