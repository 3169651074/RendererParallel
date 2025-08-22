#include <util/Matrix.cuh>

namespace renderer {
    __host__ __device__ Matrix Matrix::transpose() const {
        //创建新矩阵，元素均为0，排除原有数据干扰
        Matrix ret;
        //将第i行变为第i列
        for (size_t i = 1; i <= row; i++) { //操作的矩阵可能为列向量，不固定为5
            for (size_t j = 1; j <= col; j++) {
                ret.data[j][i] = data[i][j];
            }
        }
        return ret;
    }

    __host__ __device__ Matrix Matrix::operator*(const Matrix &right) const {
        //创建新的矩阵，行数为左矩阵的行数，列数为右矩阵的列数
        Matrix ret(row, right.col);
        //矩阵乘法，逐个元素赋值
        for (size_t i = 1; i <= ret.row; i++) {
            for (size_t j = 1; j <= ret.col; j++) {
                double sum = 0.0;
                for (size_t n = 1; n <= col; n++) {
                    sum += data[i][n] * right.data[n][j];
                }
                ret.data[i][j] = sum;
            }
        }
        return ret;
    }

    //只有4x4方阵而非列向量能够调用inverse
    __host__ __device__ Matrix Matrix::inverse() const {
        /*
         * 构造同阶单位矩阵，并将其合并到参数矩阵的右侧
         * 由于Matrix类的data限定大小为4x4，因此使用临时数组代替对象
         */
        double operateMatrix[5][9] {};
        for (size_t i = 1; i < 5; i++) {
            //将原矩阵数据填入左半部分
            for (size_t j = 1; j < 5; j++) {
                operateMatrix[i][j] = data[i][j];
            }
            //将右半部分设置为单位矩阵
            operateMatrix[i][4 + i] = 1.0;
        }

        //对合并后的矩阵进行两次消元
        if (eliminateBottomElements(operateMatrix) != 0 || eliminateTopElements(operateMatrix) != 0) {
            //矩阵不满秩，无法求逆
            return *this;
        }

        //操作矩阵的右半部分即为所求
        Matrix ret;
        for (size_t i = 1; i < 5; i++) {
            for (size_t j = 1; j < 5; j++) {
                ret.data[i][j] = operateMatrix[i][4 + j];
            }
        }
        return ret;
    }

    __host__ __device__ int Matrix::eliminateBottomElements(double matrixData[5][9]) {
        //前向消元
        for (size_t i = 1; i < 5; i++) {
            //主元选择
            double main = abs(matrixData[i][i]);
            size_t maxRow = i;
            //选取当前行及以下行最大的主元
            for (size_t p = i + 1; p < 5; p++) {
                if (abs(matrixData[p][i]) > main) {
                    main = abs(matrixData[p][i]);
                    maxRow = p;
                }
            }

            //检查最大主元是否为零
            if (floatValueNearZero(main)) {
                return 1;
            }

            if (maxRow != i) {
                //交换第maxRow行和第i行
                double tmp[9] {};
                for (size_t j = 1; j < 9; j++) {
                    tmp[j] = matrixData[maxRow][j];
                }
                for (size_t j = 1; j < 9; j++) {
                    matrixData[maxRow][j] = matrixData[i][j];
                }
                for (size_t j = 1; j < 9; j++) {
                    matrixData[i][j] = tmp[j];
                }
            }

            //操作当前行（i）的下方所有行
            for (size_t j = i + 1; j < 5; j++) {
                const double factor = matrixData[j][i] / matrixData[i][i];
                //操作一行
                for (size_t k = i; k < 9; k++) {
                    matrixData[j][k] -= factor * matrixData[i][k];
                }
            }
        }
        return 0;
    }

    __host__ __device__ int Matrix::eliminateTopElements(double matrixData[5][9]) {
        for (size_t i = 4; i >= 1; i--) {
            if (floatValueNearZero(matrixData[i][i])) {
                if (!floatValueNearZero(matrixData[i][8])) {
                    return 1; //无解
                } else {
                    return 2; //有无数个解
                }
            }

            //计算归一化系数。增广部分的所有元素都需要进行缩放
            double factor = 1 / matrixData[i][i];
            for (size_t p = i; p < 9; p++) {
                matrixData[i][p] *= factor;
            }
            for (size_t j = i - 1; j >= 1; j--) {
                factor = matrixData[j][i];
                for (size_t k = j; k < 9; k++) {
                    matrixData[j][k] -= factor * matrixData[i][k];
                }
            }
        }
        return 0;
    }

    __host__ Matrix Matrix::constructShiftMatrix(const std::array<double, 3> &shift) {
        return Matrix(4, 4, {
                1.0, 0.0, 0.0, shift[0],
                0.0, 1.0, 0.0, shift[1],
                0.0, 0.0, 1.0, shift[2],
                0.0, 0.0, 0.0, 1.0
        });
    }

    Matrix Matrix::constructScaleMatrix(const std::array<double, 3> &scale) {
        return Matrix(4, 4, {
                scale[0], 0.0, 0.0, 0.0,
                0.0, scale[1], 0.0, 0.0,
                0.0, 0.0, scale[2], 0.0,
                0.0, 0.0, 0.0, 1.0
        });
    }

    Matrix Matrix::constructRotateMatrix(double degree, int axis) {
        const double theta = degreeToRadian(degree);
        switch (axis) {
            case 0:
                return Matrix(4, 4, {
                        1.0, 0.0, 0.0, 0.0,
                        0.0, cos(theta), -sin(theta), 0.0,
                        0.0, sin(theta), cos(theta), 0.0,
                        0.0, 0.0, 0.0, 1.0
                });
            case 1:
                return Matrix(4, 4, {
                        cos(theta), 0.0, sin(theta), 0.0,
                        0.0, 1.0, 0.0, 0.0,
                        -sin(theta), 0.0, cos(theta), 0.0,
                        0.0, 0.0, 0.0, 1.0
                });
            case 2:
                return Matrix(4, 4, {
                        cos(theta), -sin(theta), 0.0, 0.0,
                        sin(theta), cos(theta), 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0
                });
            default:
                throw std::runtime_error("Invalid axis index!");
        }
    }

    Matrix Matrix::constructRotateMatrix(const std::array<double, 3> &rotate) {
        const auto mx = constructRotateMatrix(rotate[0], 0);
        const auto my = constructRotateMatrix(rotate[1], 1);
        const auto mz = constructRotateMatrix(rotate[2], 2);
        return mx * my * mz;
    }
}