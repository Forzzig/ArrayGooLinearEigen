#ifndef __BJD_H__
#define __BJD_H__
#include<LinearEigenSolver.h>

using namespace Eigen;
class BJD : public LinearEigenSolver {
public:
	int restart, gmres_size, nRestart;
	MatrixXd V, WA,/* WB, HA, HB, HAB,*/ H;
	MatrixXd Y, v;
	SparseMatrix<double> K1, tmpA;
	int cgstep;
	int batch;
	BJD(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int restart, int batch, int gmres_size);

	//用来实现求解V'(A-tauB)'AV = lam*V'(A-tauB)'BV的，暂时不用
	template<typename Derived_HA, typename Derived_HB, typename Derived_HAB>
	void generalized_RR(Derived_HA& HA, Derived_HB& HB, Derived_HAB& HAB, double tau, MatrixXd& eigenvalues, MatrixXd& eigenvectors) {
		GeneralizedSelfAdjointEigenSolver<MatrixXd> ges;
		ges.compute(HA - tau * HAB.transpose(), HAB - tau * HB);
		com_of_mul += HAB.rows() * HAB.cols() + HB.rows() * HB.cols() + 24 * HA.cols() * HA.cols() * HA.cols();
		eigenvalues = ges.eigenvalues();
		eigenvectors = ges.eigenvectors();
	}

	//设定好用来求补空间的与U对应的Y和v
	template<typename Derived_U>
	void Minv_set(Derived_U& U) {
		/*[Ki, U;]^(-1)
		   UT, 0;      * r = z
		   K1 = Ki^(-1)        */
		Y = K1 * U;
		com_of_mul += K1.nonZeros() * U.cols();
		
		v = U.transpose() * Y;
		com_of_mul += U.cols() * U.rows() * Y.cols();

		v = v.inverse();
		com_of_mul += v.cols() * v.cols() * v.cols() * 2 / 3 + 7 * v.cols() * v.cols() - v.cols() - v.cols();
	}

	//给GMRES用的，给r左乘U对应的预处理矩阵
	template<typename Derived_U, typename Derived_r, typename Derived_z>
	void Minv_mul(Derived_U& U, Derived_r& r, Derived_z& z) {
		/*[Ki, U;]^(-1)
		   UT, 0;      * r = z
		   K1 = Ki^(-1)        */
		z.topRows(K1.rows()) = K1 * r.topRows(K1.rows());
		com_of_mul += K1.nonZeros();

		z.bottomRows(U.cols()) = v * (r.bottomRows(U.cols()) + U.transpose() * z.topRows(K1.rows()));
		com_of_mul += v.rows() * v.cols() + U.cols() * U.rows();

		z.topRows(K1.rows()) -= Y * z.bottomRows(U.cols());
		com_of_mul += Y.rows() * Y.cols();
	}
	
	//左乘A对应的增广矩阵
	template<typename Derived_U, typename Derived_r, typename Derived_z>
	void A_mul(Derived_U& U, Derived_r& r, Derived_z& z) {
		z.topRows(tmpA.rows()) = tmpA * r.topRows(tmpA.rows()) + U * r.bottomRows(U.cols());
		com_of_mul += tmpA.nonZeros() + U.rows() * U.cols();
		
		z.bottomRows(U.cols()) = U.transpose() * r.topRows(tmpA.rows());
		com_of_mul += U.cols() * U.rows();
	}
	
	//左预处理GMRES，BJD特化版本
	template<typename Derived_rhs, typename Derived_ss, typename Derived_sol, typename Derived_eval>
	void L_GMRES(SparseMatrix<double>& A, SparseMatrix<double>& B, Derived_rhs& b, Derived_ss& U, Derived_sol& X, Derived_eval& lam, int m) {
		//(求解：（I-UUT）（A-λB）z = b)
		K1.resize(A.rows(), A.cols());
		K1.reserve(A.rows());
		for (int j = 0; j < A.rows(); ++j)
			K1.insert(j, j) = 1;

		MatrixXd x(A.rows() + U.cols(), 1);
		MatrixXd r(x.rows(), 1);
		MatrixXd tpw(x.rows(), 1);
		MatrixXd V(x.rows(), m);
		MatrixXd H(m + 1, m);
		MatrixXd w(x.rows(), 1);
		MatrixXd y;

		for (int i = 0; i < b.cols(); ++i) {

			//注意K逆的生成
			if (i == 0) {
				tmpA = A;
				tmpA -= lam(0, 0) * B;
			}
			else {
				tmpA -= (lam(i, 0) - lam(i - 1, 0)) * B;
			}
			com_of_mul += B.nonZeros();

			//TODO 避免0对角元,也可用其他方法
			//先取预处理矩阵为对角阵
			for (int j = 0; j < A.rows(); ++j)
				if (abs(tmpA.coeff(j, j)) < LinearEigenSolver::ORTH_TOL)
					K1.coeffRef(j, j) = 1 / LinearEigenSolver::ORTH_TOL;
				else
					K1.coeffRef(j, j) = 1 / tmpA.coeff(j, j);
			com_of_mul += A.rows() * 5;

			Minv_set(U);
			x.topRows(tmpA.rows()) = X.col(i);
			x.bottomRows(U.cols()) = MatrixXd::Zero(U.cols(), 1);

			//由于CG和GMRES不能等价，因此用CG中的cgstep近似控制一下计算量
			//这样cgstep仍然代表大矩阵乘法进行的次数，m为GMRES扩展子空间的大小，maxit为GMRES重启次数
			int maxit = cgstep / m;
			for (int it = 0; it < maxit; ++it) {

				A_mul(U, x, r);
				r.topRows(tmpA.rows()) -= b.col(i);
				tpw = -r;
				Minv_mul(U, tpw, r);

				double beta = r.norm();
				V.col(0) = r / beta;
				com_of_mul += 2 * A.rows() + 2 * U.cols();

				for (int j = 0; j < m; ++j) {
					A_mul(U, V.col(j), tpw);
					Minv_mul(U, tpw, w);
					for (int k = 0; k <= j; ++k) {
						H(k, j) = (w.transpose() * V.col(k))(0, 0);
						w -= H(k, j) * V.col(k);
					}
					H(j + 1, j) = w.norm();
					if (j < m - 1)
						V.col(j + 1) = w / H(j + 1, j);
					com_of_mul += (j + 1) * V.rows() * 2 + V.rows();

					if (H(j + 1, j) < LinearEigenSolver::ORTH_TOL) {
						H.conservativeResize(j + 2, j + 1);
						V.conservativeResize(x.rows(), j + 1);
						break;
					}
				}

				//求解y
				y = MatrixXd::Zero(H.rows(), 1);
				y(0, 0) = beta;
				for (int j = 0; j < H.rows() - 1; ++j) {
					double tmp = sqrt(H(j, j) * H(j, j) + H(j + 1, j) * H(j + 1, j));
					double s = H(j + 1, j) / tmp;
					double c = H(j, j) / tmp;
					Matrix2d omega;
					omega << c, s,
						-s, c;
					H.block(j, j, 2, H.cols() - j).applyOnTheLeft(omega);
					y(j + 1, 0) = -s * y(j, 0);
					y(j, 0) *= c;
				}
				com_of_mul += H.rows() * 30 + 2 * H.rows() * H.rows();

				for (int j = H.rows() - 2; j >= 0; --j) {
					y(j, 0) /= H(j, j);
					for (int k = j - 1; k >= 0; --k) {
						y(k, 0) -= H(k, j) * y(j, 0);
					}
				}
				com_of_mul += (H.rows() + 1) * H.rows() / 2 + 5 * H.rows();

				x += V * y.topRows(V.cols());
				com_of_mul += V.rows() * V.cols();

				if (abs(y(y.rows() - 1, 0)) < LinearEigenSolver::ORTH_TOL)
					break;
			}
			X.col(i) = x.topRows(A.rows());
		}
	}
	
	void compute();
};

#endif
