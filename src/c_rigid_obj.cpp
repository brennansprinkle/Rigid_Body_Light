#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <omp.h>
#include <random>
#include <sys/time.h>
#include <vector>
#include <stdexcept>

#include "eigen_defines.h"

namespace nb = nanobind;

struct timeval tv;
struct timezone tz;
double timeNow() {
  gettimeofday(&tv, &tz);
  int _mils = tv.tv_usec / 1000;
  int _secs = tv.tv_sec;
  return (double)_secs + ((double)_mils / 1000.0);
}

void mobilityUFRPY(real rx, real ry, real rz, real &Mxx, real &Mxy, real &Mxz,
                   real &Myy, real &Myz, real &Mzz, int i, int j, real inv_a) {
  /*
      mobilityUFRPY computes the 3x3 RPY mobility
      between blobs i and j normalized with 8 pi eta a
  */

  real fourOverThree = real(4.0) / real(3.0);

  if (i == j) {
    Mxx = fourOverThree;
    Mxy = 0;
    Mxz = 0;
    Myy = Mxx;
    Myz = 0;
    Mzz = Mxx;
  } else {
    rx = rx * inv_a;
    ry = ry * inv_a;
    rz = rz * inv_a;
    real r2 = rx * rx + ry * ry + rz * rz;
    real r = std::sqrt(r2);
    if (r < 1e-12) {
      std::cout << "ERROR: TWO BLOBS ARE OVERLAPPING OR TOO CLOSELY "
                   "POSITIONED. i="
                << i << ", j=" << j << ", r=" << r * (1.0 / inv_a) << "\n";
      exit(EXIT_FAILURE);
    }
    real invr = real(1.0) / r;
    real invr2 = invr * invr;
    real c1, c2;
    if (r >= 2) {
      c1 = real(1.0) + real(2.0) / (real(3.0) * r2);
      c2 = (real(1.0) - real(2.0) * invr2) * invr2;
      Mxx = (c1 + c2 * rx * rx) * invr;
      Mxy = (c2 * rx * ry) * invr;
      Mxz = (c2 * rx * rz) * invr;
      Myy = (c1 + c2 * ry * ry) * invr;
      Myz = (c2 * ry * rz) * invr;
      Mzz = (c1 + c2 * rz * rz) * invr;
    } else {
      c1 = fourOverThree * (real(1.0) - real(0.28125) * r); // 9/32 = 0.28125
      c2 = fourOverThree * real(0.09375) * invr;            // 3/32 = 0.09375
      Mxx = c1 + c2 * rx * rx;
      Mxy = c2 * rx * ry;
      Mxz = c2 * rx * rz;
      Myy = c1 + c2 * ry * ry;
      Myz = c2 * ry * rz;
      Mzz = c1 + c2 * rz * rz;
    }
  }
  return;
}

void mobilityUFSingleWallCorrection(real rx, real ry, real rz, real &Mxx,
                                    real &Mxy, real &Mxz, real &Myx, real &Myy,
                                    real &Myz, real &Mzx, real &Mzy, real &Mzz,
                                    int i, int j, real hj) {
  /*
      mobilityUFSingleWallCorrection computes the 3x3 mobility correction due to
     a wall between blobs i and j normalized with 8 pi eta a. This uses the
     expression from the Swan and Brady paper for a finite size particle.
      Mobility is normalize by 8*pi*eta*a.
  */
  if(hj < 0.0){
    throw std::runtime_error("A blob has its center below the wall (z<0). Cannot compute mobility- check your configuration.");
  }
  if (i == j) {
    real invZi = real(1.0) / hj;
    real invZi3 = invZi * invZi * invZi;
    real invZi5 = invZi3 * invZi * invZi;
    Mxx += -(9 * invZi - 2 * invZi3 + invZi5) / real(12.0);
    Myy += -(9 * invZi - 2 * invZi3 + invZi5) / real(12.0);
    Mzz += -(9 * invZi - 4 * invZi3 + invZi5) / real(6.0);
  } else {
    real h_hat = hj / rz;
    real invR = 1.0 / std::sqrt(rx * rx + ry * ry + rz * rz);
    real ex = rx * invR;
    real ey = ry * invR;
    real ez = rz * invR;
    real invR3 = invR * invR * invR;
    real invR5 = invR3 * invR * invR;

    real fact1 =
        -(3 * (1 + 2 * h_hat * (1 - h_hat) * ez * ez) * invR +
          2 * (1 - 3 * ez * ez) * invR3 - 2 * (1 - 5 * ez * ez) * invR5) /
        real(3.0);
    real fact2 =
        -(3 * (1 - 6 * h_hat * (1 - h_hat) * ez * ez) * invR -
          6 * (1 - 5 * ez * ez) * invR3 + 10 * (1 - 7 * ez * ez) * invR5) /
        real(3.0);
    real fact3 =
        ez *
        (3 * h_hat * (1 - 6 * (1 - h_hat) * ez * ez) * invR -
         6 * (1 - 5 * ez * ez) * invR3 + 10 * (2 - 7 * ez * ez) * invR5) *
        real(2.0) / real(3.0);
    real fact4 = ez * (3 * h_hat * invR - 10 * invR5) * real(2.0) / real(3.0);
    real fact5 = -(3 * h_hat * h_hat * ez * ez * invR + 3 * ez * ez * invR3 +
                   (2 - 15 * ez * ez) * invR5) *
                 real(4.0) / real(3.0);

    Mxx += fact1 + fact2 * ex * ex;
    Mxy += fact2 * ex * ey;
    Mxz += fact2 * ex * ez + fact3 * ex;
    Myx += fact2 * ey * ex;
    Myy += fact1 + fact2 * ey * ey;
    Myz += fact2 * ey * ez + fact3 * ey;
    Mzx += fact2 * ez * ex + fact4 * ex;
    Mzy += fact2 * ez * ey + fact4 * ey;
    Mzz += fact1 + fact2 * ez * ez + fact3 * ez + fact4 * ez + fact5;
  }
}

class CManyBodies {
  real a, dt, kBT, eta;
  // Solver parameters
  bool PC_wall = false; // use wall corrections in PC or not
  bool block_diag_PC = false;
  double M_scale;
  bool split_rand = true;
  bool PC_mat_Set = false;
  SparseM invM;
  SparseM Ninv;
  std::vector<Eigen::LLT<Matrix>> N_lu;

  // rigid coordinates
  bool cfg_set = false;
  int N_bod;
  // Base config
  std::vector<Quat> Q_n;
  std::vector<Vector> X_n;

  // Body configurations
  Matrix ref_cfg;
  int N_blb;
  bool parametersSet = false;
  SparseM K, KT, Kinv;

public:
#ifdef SINGLE_PRECISION
  static constexpr auto precision = "single";
#else
  static constexpr auto precision = "double";
#endif

  void removeMean(Matrix &cfg) {
    Vector mean = cfg.colwise().mean();
    for (int i = 0; i < cfg.rows(); ++i) {
      cfg.row(i) = cfg.row(i) - mean.transpose();
    }
  }

  void setParameters(real a, real dt, real kBT, real eta, Matrix &cfg) {
    this->a = a;
    this->dt = dt;
    this->kBT = kBT;
    this->eta = eta;
    removeMean(cfg);

    this->ref_cfg = cfg;
    this->N_blb = ref_cfg.rows();
    this->parametersSet = true;

    this->M_scale = 1.0;
  }

  void setBlkPC(bool PCtype) { block_diag_PC = PCtype; }

  void setWallPC(bool Wall) { PC_wall = Wall; }

  void setConfig(Vector X, Vector Q) {
    N_bod = X.size() / 3;
    if (!cfg_set) {
      Q_n.reserve(4 * N_bod);
      X_n.reserve(3 * N_bod);
    }

    for (int j = 0; j < N_bod; ++j) {
      Quat Q_j;
      Vector X_j(3);

      Q_j.x() = Q(4 * j + 1);
      Q_j.y() = Q(4 * j + 2);
      Q_j.z() = Q(4 * j + 3);
      Q_j.w() = Q(4 * j + 0);
      Q_j.normalize();
      if (!cfg_set) {
        Q_n.push_back(Q_j);
      } else {
        Q_n[j] = Q_j;
      }

      X_j(0) = X(3 * j + 0);
      X_j(1) = X(3 * j + 1);
      X_j(2) = X(3 * j + 2);
      if (!cfg_set) {
        X_n.push_back(X_j);
      } else {
        X_n[j] = X_j;
      }
    }
    cfg_set = true;
  }

  auto getConfig() {
    Vector Qout(4 * N_bod);
    Vector Xout(3 * N_bod);
    Quat Q_j;
    Vector X_0_j(3);
    for (int j = 0; j < N_bod; ++j) {
      Q_j = Q_n[j];

      Qout(4 * j + 1) = Q_j.x();
      Qout(4 * j + 2) = Q_j.y();
      Qout(4 * j + 3) = Q_j.z();
      Qout(4 * j + 0) = Q_j.w();

      X_0_j = X_n[j];
      Xout(3 * j + 0) = X_0_j(0);
      Xout(3 * j + 1) = X_0_j(1);
      Xout(3 * j + 2) = X_0_j(2);
    }

    return nb::make_tuple(Xout, Qout);
  }

  Matrix get_r_vecs(Vector &X_0, Quat &Q) {
    Matrix3 rotation_matrix = Q.toRotationMatrix();
    Matrix r_vectors = ref_cfg * rotation_matrix.transpose();
    Vector r_j;
    for (int j = 0; j < N_blb; ++j) {
      r_vectors.row(j) += X_0;
    }
    return r_vectors;
  }

  std::vector<real> single_body_pos(Vector &X_0, Quat &Q) {
    Matrix r_vectors = get_r_vecs(X_0, Q);
    Vector r_j;
    std::vector<real> pos;
    pos.reserve(3 * N_blb);
    for (int j = 0; j < N_blb; ++j) {
      r_j = r_vectors.row(j);
      pos.push_back(r_j(0));
      pos.push_back(r_j(1));
      pos.push_back(r_j(2));
    }
    return pos;
  }

  std::vector<real> r_vecs_from_cfg(std::vector<Vector> &Xin,
                                    std::vector<Quat> &Qin) {
    int size = N_bod * (ref_cfg.size());
    std::vector<real> pos;
    pos.reserve(size);
    std::vector<real> pos_j;
    for (int j = 0; j < N_bod; ++j) {
      pos_j = single_body_pos(Xin[j], Qin[j]);
      pos.insert(pos.end(), pos_j.begin(), pos_j.end());
      pos_j.clear();
    }
    return pos;
  }

  std::vector<real> multi_body_pos() {
    if (!cfg_set) {
      std::cout << "ERROR CONFIG NOT INITIALIZED YET!!\n";
    }
    return r_vecs_from_cfg(X_n, Q_n);
  }

  Matrix block_KTKinv(double sumr2_cfg, Matrix3 &MOI_cfg, Quat &Q_j) {
    Matrix KTKinv(6, 6);
    KTKinv.setZero();
    int N = N_blb;
    Matrix3 Ainv = (1.0 / (1.0 * N)) * Matrix::Identity(3, 3);

    Matrix3 Rot = Q_j.toRotationMatrix();
    Matrix3 D =
        (sumr2_cfg)*Matrix::Identity(3, 3) - Rot * MOI_cfg * Rot.transpose();

    double D_det = D.determinant();
    if (D_det < 1.0e-13) {
      std::cout << "ERROR K^{T}*K IS SIGULAR (is your rigid body a dimer?)\n";
      exit(EXIT_FAILURE);
    }

    Matrix3 S = D.inverse(); // = (D - C*Ainv*B)^-1;

    KTKinv.block<3, 3>(0, 0) = Ainv; // Ainv + Ainv*B*S*C*Ainv;
    // KTKinv.block<3,3>(0,3) = -1.0*Ainv*B*S;
    // KTKinv.block<3,3>(3,0) = -1.0*S*C*Ainv;
    KTKinv.block<3, 3>(3, 3) = S;

    return KTKinv;
  }

  std::tuple<SparseM, SparseM> Make_K_Kinv(std::vector<Vector> &Xin,
                                           std::vector<Quat> &Qin) {
    Matrix r_vectors;
    Vector r_k(3);
    //
    SparseM K_mat(3 * N_bod * N_blb, 6 * N_bod);
    SparseM KTKi_mat(6 * N_bod, 6 * N_bod);

    std::vector<Trip> tripletList;
    std::vector<Trip> tripletList_KTKi;
    tripletList.reserve(9 * N_blb * N_bod);
    tripletList_KTKi.reserve(6 * 6 * N_bod);

    int offset = 0;

    // sum of r_i^{T}*r_i and sum of r_i*r_i^{T}
    // where r_i is in Ref config (for (KT*K)^-1)
    double sumr2_cfg = ref_cfg.squaredNorm();
    Matrix3 MOI_cfg = Matrix3::Zero();
    for (int k = 0; k < N_blb; ++k) {
      r_k = ref_cfg.row(k);
      MOI_cfg += r_k * r_k.transpose();
    }

    Matrix KTKi_block(6, 6);

    for (int j = 0; j < N_bod; ++j) {

      r_vectors = get_r_vecs(Xin[j], Qin[j]);

      // Set blocks of (K^T*K)^-1
      KTKi_block = block_KTKinv(sumr2_cfg, MOI_cfg, Qin[j]);

      for (int rw = 0; rw < 6; ++rw) {
        for (int cl = 0; cl < 6; ++cl) {
          tripletList_KTKi.push_back(
              Trip(6 * j + rw, 6 * j + cl, KTKi_block(rw, cl)));
        }
      }

      // set blocks of K
      for (int k = 0; k < N_blb; ++k) {
        tripletList.push_back(Trip(offset + 3 * k + 0, 6 * j + 0, 1.0));
        tripletList.push_back(Trip(offset + 3 * k + 1, 6 * j + 1, 1.0));
        tripletList.push_back(Trip(offset + 3 * k + 2, 6 * j + 2, 1.0));

        r_k = r_vectors.row(k) - Xin[j].transpose();

        tripletList.push_back(Trip(offset + 3 * k + 0, 6 * j + 4, r_k(2)));
        tripletList.push_back(Trip(offset + 3 * k + 0, 6 * j + 5, -r_k(1)));
        tripletList.push_back(Trip(offset + 3 * k + 1, 6 * j + 5, r_k(0)));

        tripletList.push_back(Trip(offset + 3 * k + 1, 6 * j + 3, -r_k(2)));
        tripletList.push_back(Trip(offset + 3 * k + 2, 6 * j + 3, r_k(1)));
        tripletList.push_back(Trip(offset + 3 * k + 2, 6 * j + 4, -r_k(0)));
      }
      offset += 3 * N_blb;
    }
    K_mat.setFromTriplets(tripletList.begin(), tripletList.end());

    KTKi_mat.setFromTriplets(tripletList_KTKi.begin(), tripletList_KTKi.end());

    SparseM KI = (KTKi_mat * K_mat.transpose()).pruned();

    return std::make_tuple(K_mat, KI);
  }

  void set_K_mats() {
    if (!cfg_set) {
      std::cout << "ERROR CONFIG NOT INITIALIZED YET!!\n";
    }

    std::tie(K, Kinv) = Make_K_Kinv(X_n, Q_n);
    KT = K.transpose();
  }

  Vector K_x_U(const Vector &U) { return K * U; }

  Vector Kinv_x_V(const Vector &V) { return Kinv * V; }

  Vector KTinv_x_F(const Vector &F) { return Kinv.transpose() * F; }

  Vector KT_x_Lam(const Vector &Lam) { return KT * Lam; }

  /////////////////   preconditioner/solver functions /////////////////
  template <class AVector> Matrix rotne_prager_tensor(AVector &r_vectors) {

    real norm_fact_f = real(1.0) / (8.0 * M_PI * eta * a);

    int N = r_vectors.size();
    int Nparts = N / 3;

    real inv_a = real(1.0) / a;

    real rx, ry, rz;

    real Mxx, Mxy, Mxz;
    real Myx, Myy, Myz;
    real Mzx, Mzy, Mzz;

    Matrix Mob(N, N);

    for (int i = 0; i < Nparts; ++i) {
      for (int j = i; j < Nparts; ++j) {
        rx = r_vectors[3 * i + 0] - r_vectors[3 * j + 0];
        ry = r_vectors[3 * i + 1] - r_vectors[3 * j + 1];
        rz = r_vectors[3 * i + 2] - r_vectors[3 * j + 2];

        mobilityUFRPY(rx, ry, rz, Mxx, Mxy, Mxz, Myy, Myz, Mzz, i, j, inv_a);
        Myx = Mxy;
        Mzx = Mxz;
        Mzy = Myz;
        if (PC_wall) {
          mobilityUFSingleWallCorrection(rx / a, ry / a,
                                         (rz + 2 * r_vectors[3 * j + 2]) / a,
                                         Mxx, Mxy, Mxz, Myx, Myy, Myz, Mzx, Mzy,
                                         Mzz, i, j, r_vectors[3 * j + 2] / a);
        }

        Matrix3 Block{{Mxx, Mxy, Mxz}, {Myx, Myy, Myz}, {Mzx, Mzy, Mzz}};

        Mob.block<3, 3>(3 * i, 3 * j) = Block;
        if (j != i) {
          Mob.block<3, 3>(3 * j, 3 * i) = Block.transpose();
        }
      }
    }

    Mob *= norm_fact_f;

    return Mob;
  }

  SparseM Block_diag_invM() {

    int Blk_sz = 3 * N_blb;

    SparseM Blk_Mob(Blk_sz * N_bod, Blk_sz * N_bod);
    Matrix Mob(Blk_sz, Blk_sz);
    Matrix Minv(Blk_sz, Blk_sz);

    std::vector<Trip> tripletList;
    tripletList.reserve(Blk_sz * Blk_sz * N_bod);

    for (int i = 0; i < N_bod; ++i) {
      std::vector<real> r_vectors = single_body_pos(X_n[i], Q_n[i]);
      Mob = rotne_prager_tensor(r_vectors);
      Minv = Mob.inverse();

      for (int rw = 0; rw < Blk_sz; ++rw) {
        for (int cl = 0; cl < Blk_sz; ++cl) {
          tripletList.push_back(
              Trip(Blk_sz * i + rw, Blk_sz * i + cl, Minv(rw, cl)));
        }
      }
    }
    Blk_Mob.setFromTriplets(tripletList.begin(), tripletList.end());

    return Blk_Mob;
  }

  template <class AVector> SparseM diag_invM(AVector &r_vectors) {

    real norm_fact_f = real(8.0 * M_PI * eta * a);

    int N = r_vectors.size();
    int Nparts = N / 3;

    real inv_a = real(1.0) / a;

    real rx = 0;
    real ry = 0;
    real rz = 0;

    real Mxx, Mxy, Mxz;
    real Myx, Myy, Myz;
    real Mzx, Mzy, Mzz;

    std::vector<Trip> tripletList;
    tripletList.reserve(Nparts * 3 * 3);

    for (int i = 0; i < Nparts; ++i) {
      int j = i;

      mobilityUFRPY(rx, ry, rz, Mxx, Mxy, Mxz, Myy, Myz, Mzz, i, j, inv_a);
      Myx = Mxy;
      Mzx = Mxz;
      Mzy = Myz;
      if (PC_wall) {
        mobilityUFSingleWallCorrection(
            rx / a, ry / a, (rz + 2 * r_vectors[3 * j + 2]) / a, Mxx, Mxy, Mxz,
            Myx, Myy, Myz, Mzx, Mzy, Mzz, i, j, r_vectors[3 * j + 2] / a);
      }

      Matrix3 Block{{Mxx, Mxy, Mxz}, {Myx, Myy, Myz}, {Mzx, Mzy, Mzz}};

      Matrix3 Minv = Block.inverse();

      tripletList.push_back(Trip(i * 3, j * 3, Minv(0, 0)));
      tripletList.push_back(Trip(i * 3, j * 3 + 1, Minv(0, 1)));
      tripletList.push_back(Trip(i * 3, j * 3 + 2, Minv(0, 2)));

      tripletList.push_back(Trip(i * 3 + 1, j * 3, Minv(1, 0)));
      tripletList.push_back(Trip(i * 3 + 1, j * 3 + 1, Minv(1, 1)));
      tripletList.push_back(Trip(i * 3 + 1, j * 3 + 2, Minv(1, 2)));

      tripletList.push_back(Trip(i * 3 + 2, j * 3, Minv(2, 0)));
      tripletList.push_back(Trip(i * 3 + 2, j * 3 + 1, Minv(2, 1)));
      tripletList.push_back(Trip(i * 3 + 2, j * 3 + 2, Minv(2, 2)));
    }
    SparseM mat(N, N);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    mat *= norm_fact_f;

    return mat;
  }

  SparseM PC_invM() {
    if (!block_diag_PC) {
      std::vector<real> r_vectors = multi_body_pos();
      return diag_invM(r_vectors);
    } else {
      return Block_diag_invM();
    }
  }

  std::vector<Eigen::LLT<Matrix>> get_blk_diag_lu(SparseM &Ninv) {
    Matrix Block(6, 6);

    std::vector<Eigen::LLT<Matrix>> N_lu;
    N_lu.reserve(N_bod * 6 * 6);

    for (int i = 0; i < N_bod; ++i) {
      Block = Ninv.block(6 * i, 6 * i, 6, 6);
      Eigen::LLT<Matrix> lu(Block);
      N_lu.push_back(lu);
    }

    return N_lu;
  }

  template <class AVector> void test_PC(AVector &Lambda, AVector &U) {

    SparseM invM = PC_invM();

    Eigen::SimplicialLLT<SparseM> chol(invM);
    Vector Slip = chol.solve(Lambda) - K * U;
    Vector F = -KT * Lambda;

    Vector IN(3 * N_bod * N_blb + 6 * N_bod);
    IN << Slip, F;
    Vector OUT = apply_PC(IN);
    Vector Lpc = OUT.head(3 * N_bod * N_blb);
    Vector Upc = OUT.tail(6 * N_bod);

    Lpc *= (1.0 / M_scale);

    std::cout << "error Lambda: " << (Lpc - Lambda).norm() << "\n";
    std::cout << "error U: " << (Upc - U).norm() << "\n";
  }

  Vector apply_PC(const Vector &IN) {

    if (not PC_mat_Set) {
      invM = PC_invM();
      Ninv = (KT * invM * K).pruned();
      N_lu = get_blk_diag_lu(Ninv);
      PC_mat_Set = true;
    }

    Vector Slip = IN.head(3 * N_bod * N_blb);
    Vector F = IN.tail(6 * N_bod);

    Vector RHS = -F - KT * (invM * Slip);

    Vector b(6);
    Vector U(6 * N_bod);
    for (int i = 0; i < N_bod; ++i) {
      b = RHS.segment<6>(6 * i);
      U.segment<6>(6 * i) = N_lu[i].solve(b);
    }

    Vector Lambda = M_scale * (invM * (Slip + K * U));

    Vector Out(3 * N_bod * N_blb + 6 * N_bod);
    Out << Lambda, U;

    return Out;
  }

  DiagM make_damp_mat(const std::vector<real> &r_vectors) {
    /*
     matrix used to regularise the mobility matrix to allow for particle-wall
     overlaps see Balboa Usabiaga, Delmotte, and Donev 2017 appendix A
   */
    int N = r_vectors.size();
    int Nparts = N / 3;
    Vector B(N);
    double d_ii;

    for (int i = 0; i < Nparts; ++i) {
      if (r_vectors[3 * i + 2] >= a) {
        d_ii = 1.0;
      } else {
        d_ii = r_vectors[3 * i + 2] / a;
      }
      for (int k = 0; k < 3; ++k) {
        B(3 * i + k) = d_ii;
      }
    }
    return B.asDiagonal();
  }

  Vector apply_M(const Vector &F, const std::vector<real> &r_vectors) {
    /*
     returns U = B(r) * { M(r) * [ B(r) * F ] }
     F and r are 3*Nblob*Nbody vectors
     M is the blob-blob mobilities
    */
    int sz = r_vectors.size();
    Vector U(sz);
    U.setZero();
    Matrix M = rotne_prager_tensor(r_vectors);
    if (PC_wall) {
      DiagM B = make_damp_mat(r_vectors);
      U = B * (M * (B * F));
    } else {
      U = M * F;
    }

    return U;
  }

  Vector M_half_W() {
    std::vector<real> r_vectors = multi_body_pos();
    int sz = 3 * N_bod * N_blb;
    Vector W = rand_vector(sz);
    Vector Out(sz);

    Matrix Mob = rotne_prager_tensor(r_vectors);
    DiagM B = make_damp_mat(r_vectors);
    Matrix M = B * Mob * B;
    Eigen::LLT<Matrix> chol(M);
    Matrix L = chol.matrixL();
    Out = (L * W);

    return Out;
  }

  /////////////////   Dynamics/time integration /////////////////

  Quat Q_from_Om(Vector &Om) {
    // ...
    Quat Q_rot = Quat::Identity();
    double Om_norm = Om.norm();
    Q_rot.w() = cos(Om_norm / 2.0);
    if (Om_norm > 1.0e-10) {
      Q_rot.vec() = (sin(Om_norm / 2.0) / Om_norm) * Om;
    }
    Q_rot.normalize();
    return Q_rot;
  }

  std::tuple<std::vector<Quat>, std::vector<Vector>> update_X_Q(Vector &U) {
    // TODO the input should not be called U since it has the units of X
    std::vector<Quat> Qin(Q_n);
    std::vector<Vector> Xin(X_n);

    Quat Q_rot;
    Vector U_j(3);
    Vector Om_j(3);

    for (int j = 0; j < N_bod; ++j) {
      U_j = U.segment<3>(6 * j);
      Om_j = U.segment<3>(6 * j + 3);
      Q_rot = Q_from_Om(Om_j);
      Qin[j] = Q_rot * Qin[j];
      Qin[j].normalize();
      Xin[j] = Xin[j] + U_j;
    }

    return std::make_tuple(Qin, Xin);
  }

  std::tuple<Matrix, Matrix> update_X_Q_out(Vector &U) {
    std::vector<Quat> Qin;
    std::vector<Vector> Xin;
    std::tie(Qin, Xin) = update_X_Q(U);

    Matrix Qout(N_bod, 4);
    Matrix Xout(N_bod, 3);
    for (int j = 0; j < N_bod; ++j) {
      Qout(j, 0) = Qin[j].w();
      Qout(j, 1) = Qin[j].x();
      Qout(j, 2) = Qin[j].y();
      Qout(j, 3) = Qin[j].z();
      //
      Xout.row(j) = Xin[j];
    }
    return std::make_tuple(Qout, Xout);
  }

  Vector rand_vector(int N) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0.0, 1.0);

    Vector W = Vector::Zero(N);
    for (int k = 0; k < N; ++k) {
      W[k] = distribution(generator);
    }

    return W;
  }

  Vector KTinv_RFD() {

    double delta = 1.0e-4;

    Vector W = rand_vector(6 * N_bod);

    std::vector<Quat> Qp;
    std::vector<Vector> Xp;
    std::vector<Quat> Qm;
    std::vector<Vector> Xm;

    Vector Win = ((delta / 2.0) * W);
    std::tie(Qp, Xp) = update_X_Q(Win);
    Win *= -1.0;
    std::tie(Qm, Xm) = update_X_Q(Win);

    SparseM Kp, Kinvp, Km, Kinvm;
    std::tie(Kp, Kinvp) = Make_K_Kinv(Xp, Qp);
    std::tie(Km, Kinvm) = Make_K_Kinv(Xm, Qm);

    Vector out = (1.0 / delta) * Kinvp.transpose() * W -
                 (1.0 / delta) * Kinvm.transpose() * W;

    return (KT * out);
  }

  Vector M_RFD() {

    double delta = 1.0e-4;

    int sz = 3 * N_bod * N_blb;
    Vector W = rand_vector(sz);

    Vector UOM = Kinv * W;

    std::vector<Quat> Qp;
    std::vector<Vector> Xp;
    std::vector<Quat> Qm;
    std::vector<Vector> Xm;

    Vector Win = ((delta / 2.0) * UOM);
    std::tie(Qp, Xp) = update_X_Q(Win);
    std::vector<real> r_vec_p = r_vecs_from_cfg(Xp, Qp);
    Win *= -1.0;
    std::tie(Qm, Xm) = update_X_Q(Win);
    std::vector<real> r_vec_m = r_vecs_from_cfg(Xm, Qm);

    Vector Mp = apply_M(W, r_vec_p);
    Vector Mm = apply_M(W, r_vec_m);

    Vector out = (1.0 / delta) * (Mp - Mm);

    return out;
  }

  template <class AVector> auto M_RFD_cfgs(AVector &U, double delta) {

    int sz = 3 * N_bod * N_blb;
    Vector W = rand_vector(sz);

    std::vector<Quat> Qp;
    std::vector<Vector> Xp;
    std::vector<Quat> Qm;
    std::vector<Vector> Xm;

    Vector Win = ((delta / 2.0) * U);
    std::tie(Qp, Xp) = update_X_Q(Win);
    std::vector<real> r_vec_p = r_vecs_from_cfg(Xp, Qp);
    Win *= -1.0;
    std::tie(Qm, Xm) = update_X_Q(Win);
    std::vector<real> r_vec_m = r_vecs_from_cfg(Xm, Qm);

    return std::make_tuple(r_vec_p, r_vec_m);
  }

  template <class AVector> Vector M_RFD_from_U(AVector &U, AVector &W) {

    double delta = 1.0e-3;

    std::vector<Quat> Qp;
    std::vector<Vector> Xp;
    std::vector<Quat> Qm;
    std::vector<Vector> Xm;

    Vector Win = ((delta / 2.0) * U);
    std::tie(Qp, Xp) = update_X_Q(Win);
    std::vector<real> r_vec_p = r_vecs_from_cfg(Xp, Qp);
    Win *= -1.0;
    std::tie(Qm, Xm) = update_X_Q(Win);
    std::vector<real> r_vec_m = r_vecs_from_cfg(Xm, Qm);

    Vector Mp = apply_M(W, r_vec_p);
    Vector Mm = apply_M(W, r_vec_m);

    Vector out = (1.0 / delta) * (Mp - Mm);

    return out;
  }

  template <class AVector> Vector KT_RFD_from_U(AVector &U, AVector &W) {

    double delta = 1.0e-3;

    std::vector<Quat> Qp;
    std::vector<Vector> Xp;
    std::vector<Quat> Qm;
    std::vector<Vector> Xm;

    Vector Win = ((delta / 2.0) * U);
    std::tie(Qp, Xp) = update_X_Q(Win);
    Win *= -1.0;
    std::tie(Qm, Xm) = update_X_Q(Win);

    SparseM Kp, Kinvp, Km, Kinvm;
    std::tie(Kp, Kinvp) = Make_K_Kinv(Xp, Qp);
    std::tie(Km, Kinvm) = Make_K_Kinv(Xm, Qm);

    Vector out = (1.0 / delta) * (Kp.transpose() * W - Km.transpose() * W);

    return out;
  }

  void evolve_X_Q(Vector &U) {
    std::vector<Quat> Qnext;
    std::vector<Vector> Xnext;

    U *= dt;

    std::tie(Qnext, Xnext) = update_X_Q(U);

    Q_n = Qnext;
    X_n = Xnext;

    set_K_mats();
    PC_mat_Set = false;
  }

  void evolve_X_Q_RFD(Vector &U) {
    // U here should have units of displacements (already multiplied by dt or
    // delta)
    std::vector<Quat> Qnext;
    std::vector<Vector> Xnext;

    std::tie(Qnext, Xnext) = update_X_Q(U);

    Q_n = Qnext;
    X_n = Xnext;

    set_K_mats();
    PC_mat_Set = true;
  }

  Vector Test_Mhalf(int N) {
    Vector error(N);
    std::vector<real> r_vectors = multi_body_pos();
    Matrix M = rotne_prager_tensor(r_vectors);
    Matrix M_rand(3 * N_bod * N_blb, 3 * N_bod * N_blb);
    M_rand.setZero();
    Matrix Dif(3 * N_bod * N_blb, 3 * N_bod * N_blb);

    double M_scale = M.norm();

    for (int i = 0; i < N; ++i) {
      if (i % (N / 10) == 0) {
        std::cout << "itteration: " << i << "\n";
      }
      Vector M_half_W1 = M_half_W();
      M_rand += M_half_W1 * M_half_W1.transpose();
      Dif = ((1.0 / (double)i) * M_rand - M);
      error(i) = (1.0 / M_scale) * Dif.norm();
    }
    return error;
  }

  auto RHS_and_Midpoint(RefVector &Slip, RefVector &Force) {
    double t, elapsed;

    std::vector<real> r_vecs;

    if (kBT > 1e-10) {
      std::vector<Quat> Qhalf;
      std::vector<Vector> Xhalf;
      Vector M_RFD_vec;
      Vector BI;

      // Make Brownian increment for predictor and corrector steps
      t = timeNow();
      Vector M_half_W1 = M_half_W();
      elapsed = timeNow() - t;
      printf("Root time = %g\n", elapsed);
      Vector M_half_W2;
      if (split_rand) {
        M_half_W2 = M_half_W();
      }

      // Make M_RFD
      std::cout << "Before RFD\n";
      M_RFD_vec = M_RFD();
      std::cout << "After RFD\n";

      // Set predictor velocity
      double c_1, c_2;
      if (split_rand) {
        c_1 = 2.0 * std::sqrt((kBT / dt));
        c_2 = std::sqrt((kBT / dt));
        BI = c_2 * (M_half_W1 - M_half_W2);
      } else {
        c_1 = std::sqrt(2.0 * (kBT / dt));
        c_2 = std::sqrt(2.0 * (kBT / dt));
        BI = c_2 * (M_half_W1);
      }
      Vector BI_half = c_1 * M_half_W1;
      Vector UOm_half = (dt / 2.0) * Kinv * BI_half;

      std::tie(Qhalf, Xhalf) = update_X_Q(UOm_half);
      r_vecs = r_vecs_from_cfg(Xhalf, Qhalf);

      set_K_mats();

      // Make RHS for final solve
      Slip -= ((kBT * M_RFD_vec) + BI);
    } else {
      // print that we are not using Brownian motion
      std::cout << "No Brownian terms\n";
    }

    // Slip *= -1.0;
    Force *= -1.0;

    Vector RHS(3 * N_bod * N_blb + 6 * N_bod);
    RHS << Slip, Force;

    return RHS;
  }

  auto get_K() {
    if (!cfg_set) {
      std::cout << "ERROR CONFIG NOT INITIALIZED YET!!\n";
    }

    return K;
  }

  auto get_Kinv() {
    if (!cfg_set) {
      std::cout << "ERROR CONFIG NOT INITIALIZED YET!!\n";
    }

    return Kinv;
  }

private:
};

NB_MODULE(c_rigid, m) {
  m.doc() = "Rigid code";

  nb::class_<CManyBodies> cls(m, "CManyBodies");
  cls.def(nb::init<>())
      .def("getConfig", &CManyBodies::getConfig,
           "get the X and Q vectors for the current position")
      .def("setParameters", &CManyBodies::setParameters,
           "Set parameters for the module")
      .def("setBlkPC", &CManyBodies::setBlkPC, "set PC type")
      .def("setWallPC", &CManyBodies::setWallPC, "use wall corrections")
      .def("set_K_mats", &CManyBodies::set_K_mats,
           "Set the K,K^T,K^-1 matrices for the module")
      .def("K_x_U", &CManyBodies::K_x_U, "Multiply K by U", nb::arg("U"))
      .def("KT_x_Lam", &CManyBodies::KT_x_Lam, "Multiply K^T by lambda",
           nb::arg("lambda"))
      .def("multi_body_pos", &CManyBodies::multi_body_pos,
           "Get the blob positions")
      .def("apply_PC", &CManyBodies::apply_PC, "apply for PC")
      .def("setConfig", &CManyBodies::setConfig,
           "Set the X and Q vectors for the current position", nb::arg("X"),
           nb::arg("Q"))
      .def("get_K", &CManyBodies::get_K, "get K")
      .def("get_Kinv", &CManyBodies::get_Kinv, "get Kinv")
      .def("evolve_X_Q", &CManyBodies::evolve_X_Q, "evolve rigid bodies",
           nb::arg("U"))
      .def("apply_M", &CManyBodies::apply_M, "mobility matrix mult", nb::arg("F"), nb::arg("r_vecs"))
      .def_prop_ro_static(
          "precision", [](nb::object) { return CManyBodies::precision; },
          R"pbdoc(Compilation precision, a string holding either single or double.)pbdoc");
}