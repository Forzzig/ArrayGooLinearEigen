#ifndef __ITERRITZ_H__
#define __ITERRITZ_H__
#include<LinearEigenSolver.h>

using namespace Eigen;
class IterRitz : public LinearEigenSolver {
public:
	MatrixXd X, V, Lam, P;
	int q, r, cgstep;
	MatrixXd CA, CB, CAB, CBA;

	ConjugateGradient<SparseMatrix<double, RowMajor, __int64>, Lower | Upper> linearsolver;

	//X������V��ͬ
	template<typename Derived_v, typename Derived_x>
	void refine(double lam, Derived_v& V, Derived_x& X) {
		MatrixXd C(CA.rows(), CA.cols());
		C = CA - lam * (CAB + CBA) + (lam * lam) * CB;
		com_of_mul += 2 * V.cols() * V.cols();

		JacobiSVD<MatrixXd, NoQRPreconditioner> SVDsolver;
		SVDsolver.compute(C, ComputeThinV);
		com_of_mul += 24 * V.cols() * V.cols() * V.cols();

		//�������ֵΪ0������Ӧ���Ѿ����������ټ�������������ֵ��
		int p = C.cols();
		while (SVDsolver.singularValues()(p - 1, 0) < EIGTOL)
			--p;
		//ȡ��Ӧ����С����ֵ����������
		X.noalias() = V * SVDsolver.matrixV().middleCols(p - X.cols(), X.cols());
		com_of_mul += A.rows() * V.cols();
	}

	IterRitz(SparseMatrix<double, RowMajor, __int64>& A, SparseMatrix<double, RowMajor, __int64>& B, int nev, int cgstep, int q, int r);
	void compute();
};
#endif
