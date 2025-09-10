
#ifndef C_RIGID_EIGEN_TYPEDEFS_H
#define C_RIGID_EIGEN_TYPEDEFS_H

#ifndef DOUBLE_PRECISION
#define SINGLE_PRECISION
#endif

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef Eigen::VectorXi IVector;
typedef Eigen::Ref<IVector> IRefVector;
typedef Eigen::Ref<const IVector> CstIRefVector;

#ifdef SINGLE_PRECISION
using real = float;
typedef Eigen::VectorXf Vector;
typedef Eigen::Ref<Vector> RefVector;
typedef Eigen::Ref<const Vector> CstRefVector;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::Ref<Matrix, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
    RefMatrix;
typedef Eigen::Quaternionf Quat;
typedef Eigen::Matrix3f Matrix3;
#else
using real = double;
typedef Eigen::VectorXd Vector;
typedef Eigen::Ref<Vector> RefVector;
typedef Eigen::Ref<const Vector> CstRefVector;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::Ref<Matrix, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
    RefMatrix;
typedef Eigen::Quaterniond Quat;
typedef Eigen::Matrix3d Matrix3;
#endif

typedef Eigen::Triplet<real> Trip;
typedef Eigen::SparseMatrix<real> SparseM;
typedef Eigen::DiagonalMatrix<real, Eigen::Dynamic> DiagM;
typedef Eigen::Triplet<real> Trip_d;

#endif // C_RIGID_EIGEN_TYPEDEFS_H