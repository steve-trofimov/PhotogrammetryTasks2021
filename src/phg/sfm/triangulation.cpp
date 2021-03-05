#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    double x0 = ms[0][0];
    double y0 = ms[0][1];
    double x1 = ms[1][0];
    double y1 = ms[1][1];

    Eigen::Matrix4d A;
    A.resize(2 * count, 4);
    for (int i = 0; i < count; i++) {
        auto row0 = ms[i][0] * Ps[i].row(2) - ms[i][2] * Ps[i].row(0);
        auto row1 = ms[i][1] * Ps[i].row(2) - ms[i][2] * Ps[i].row(1);

        A.row(i * 2) << row0(0), row0(1), row0(2), row0(3);
        A.row(i * 2 + 1) << row1(0), row1(1), row1(2), row1(3);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXd nullVecSvd = svd.matrixV().col(svd.matrixV().cols() - 1);
    return {nullVecSvd[0], nullVecSvd[1], nullVecSvd[2], nullVecSvd[3]};
}
