#ifndef __BJD_H__
#define __BJD_H__
#include<Eigen\Sparse>
#include<Eigen\Dense>
#include<EigenResult.h>
#include<iostream>
#include<LinearEigenSolver.h>

using namespace Eigen;
class BJD : public LinearEigenSolver {
public:
	int restart, gmres_size;
	MatrixXd V, WA, WB, HA, HB, HAB, H;
	MatrixXd Y, v;
	SparseMatrix<double> K1, tmpA;
	int cgstep;
	int batch;
	BJD(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int restart, int batch, int gmres_size);

	template<typename Derived_HA, typename Derived_HB, typename Derived_HAB>
	void generalized_RR(Derived_HA& HA, Derived_HB& HB, Derived_HAB& HAB, double tau, MatrixXd& eigenvalues, MatrixXd& eigenvectors);

	template<typename Derived_U>
	void Minv_set(Derived_U& U);

	template<typename Derived_U, typename Derived_r, typename Derived_z>
	void Minv_mul(Derived_U& U, Derived_r& r, Derived_z& z);

	template<typename Derived_U, typename Derived_r, typename Derived_z>
	void A_mul(Derived_U& U, Derived_r& r, Derived_z& z);
	
	template<typename Derived_rhs, typename Derived_ss, typename Derived_sol, typename Derived_eval>
	void L_GMRES(SparseMatrix<double>& A, SparseMatrix<double>& B, Derived_rhs& b, Derived_ss& U, Derived_sol& X, Derived_eval& lam, int m);

	void compute();
};

//����ʵ�����V'(A-tauB)'AV = lam*V'(A-tauB)'BV�ģ���ʱ����
template<typename Derived_HA, typename Derived_HB, typename Derived_HAB>
void BJD::generalized_RR(Derived_HA& HA, Derived_HB& HB, Derived_HAB& HAB, double tau, MatrixXd& eigenvalues, MatrixXd& eigenvectors){
	GeneralizedSelfAdjointEigenSolver<MatrixXd> ges;
	ges.compute(HA - tau * HAB.transpose(), HAB - tau * HB);
	eigenvalues = ges.eigenvalues();
	eigenvectors = ges.eigenvectors();
}

//�趨�������󲹿ռ����U��Ӧ��Y��v
template<typename Derived_U>
void BJD::Minv_set(Derived_U& U) {
	/*[Ki, U;]^(-1)
	   UT, 0;      * r = z
	   K1 = Ki^(-1)        */
	Y = K1 * U;
	v = U.transpose() * Y;
	v = v.inverse();
}

//��GMRES�õģ���r���U��Ӧ��Ԥ�������
template<typename Derived_U, typename Derived_r, typename Derived_z>
void BJD::Minv_mul(Derived_U& U, Derived_r& r, Derived_z& z) {
	/*[Ki, U;]^(-1)
	   UT, 0;      * r = z
	   K1 = Ki^(-1)        */
	z.block(0, 0, K1.rows(), 1) = K1 * r.block(0, 0, K1.rows(), 1);
	z.block(K1.rows(), 0, U.cols(), 1) = v * (r.block(K1.rows(), 0, U.cols(), 1) + U.transpose() * z.block(0, 0, K1.rows(), 1));
	z.block(0, 0, K1.rows(), 1) -= Y * z.block(K1.rows(), 0, U.cols(), 1);
}

//���A��Ӧ���������
template<typename Derived_U, typename Derived_r, typename Derived_z>
void BJD::A_mul(Derived_U& U, Derived_r& r, Derived_z& z) {
	z.block(0, 0, tmpA.rows(), 1) = tmpA * r.block(0, 0, tmpA.rows(), 1) + U * r.block(tmpA.rows(), 0, U.cols(), 1);
	z.block(tmpA.rows(), 0, U.cols(), 1) = U.transpose() * r.block(0, 0, tmpA.rows(), 1);
}

//��Ԥ����GMRES��ΪBJD�ػ�
template<typename Derived_rhs, typename Derived_ss, typename Derived_sol, typename Derived_eval>
void BJD::L_GMRES(SparseMatrix<double>& A, SparseMatrix<double>& B, Derived_rhs& b, Derived_ss& U, Derived_sol& X, Derived_eval& lam, int m) {
	//(��⣺��I-UUT����A-��B��z = b)
	K1.resize(A.rows(), A.cols());
	K1.reserve(A.rows());
	for (int j = 0; j < A.rows(); ++j)
		K1.insert(j, j) = 1;
	tmpA = A;

	//ע��K�������
	/*if (i == 0) {
			tmpA -= lam(0, 0) * B;
		}
		else {
			tmpA -= (lam(i, 0) - lam(i - 1, 0)) * B;
		}*/
	//��ȡԤ�������Ϊ�Խ���
	for (int j = 0; j < A.rows(); ++j)
		K1.coeffRef(j, j) = 1 / tmpA.coeff(j, j);
	Minv_set(U);

	for (int i = 0; i < b.cols(); ++i) {

		MatrixXd x(tmpA.rows() + U.cols(), 1);
		x.block(0, 0, tmpA.rows(), 1) = X.col(i);
		x.block(tmpA.rows(), 0, U.cols(), 1) = MatrixXd::Zero(U.cols(), 1);

		//����CG��GMRES���ܵȼۣ������CG�е�cgstep���ƿ���һ�¼�����
		//����cgstep��Ȼ��������˷����еĴ�����mΪGMRES��չ�ӿռ�Ĵ�С��maxitΪGMRES��������
		int maxit = cgstep / m;
		for (int it = 0; it < maxit; ++it) {

			MatrixXd r(x.rows(), 1);
			A_mul(U, x, r);
			r.block(0, 0, tmpA.rows(), 1) -= b.col(i);
			MatrixXd tpw(x.rows(), 1);
			tpw = -r;
			Minv_mul(U, tpw, r);

			MatrixXd V(x.rows(), m);
			MatrixXd H(m + 1, m);
			double beta = r.norm();
			V.col(0) = r / beta;
			MatrixXd w(x.rows(), 1);

			/*coutput << "iteration-----------------------" << endl << it << endl;
			coutput << "r0-----------------------" << endl << r << endl;
			coutput << "v1-----------------------" << endl << V.col(0) << endl;

			coutput << "beta-----------------------" << endl << beta << endl;*/
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
				if (H(j + 1, j) < LinearEigenSolver::ORTH_TOL) {
					H.conservativeResize(j + 2, j + 1);
					V.conservativeResize(x.rows(), j + 1);
					break;
				}
				/*coutput << "w-----------------------" << endl << w << endl;*/
			}
			
			/*coutput << "H-----------------------" << endl << H << endl;*/

			//���y
			MatrixXd y = MatrixXd::Zero(H.rows(), 1);
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
			for (int j = H.rows() - 2; j >= 0; --j) {
				y(j, 0) /= H(j, j);
				for (int k = j - 1; k >= 0; --k) {
					y(k, 0) -= H(k, j) * y(j, 0);
				}
			}
			x += V * y.block(0, 0, V.cols(), 1);
			/*coutput << "y-----------------------" << endl << y << endl;*/
			if (abs(y(y.rows() - 1, 0)) < LinearEigenSolver::ORTH_TOL)
				break;
		}
		X.col(i) = x.block(0, 0, A.rows(), 1);
	}
}
#endif
