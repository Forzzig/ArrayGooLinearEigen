#ifndef __REFINERITZ_H__
#define __REFINERITZ_H__
#include<LinearEigenSolver.h>

//#define use_P
#define use_refine
//#define use_X
//#define use_Xr

using namespace Eigen;
class refineRitz : public LinearEigenSolver {
public:
	MatrixXd X, V, Lam, P;
	int q, r, cgstep, Vsize;
	double ratio;
	MatrixXd CA, CB, CAB, CBA;

	ConjugateGradient<SparseMatrix<double, RowMajor, __int64>, Lower | Upper> linearsolver;
	
	//TODO 自定义预优
	//BiCGSTAB<SparseMatrix<double, RowMajor, __int64>> linearsolver2;

	//X必须与V不同
	template<typename Derived_v, typename Derived_x>
	void refine(double lam, Derived_v& V, Derived_x& X) {
		MatrixXd C(CA.rows(), CA.cols());
		C = CA - lam * (CAB + CBA) + (lam * lam) * CB;
		com_of_mul += 2 * V.cols() * V.cols();

		JacobiSVD<MatrixXd, NoQRPreconditioner> SVDsolver;
		SVDsolver.compute(C, ComputeThinV);
		com_of_mul += 24 * V.cols() * V.cols() * V.cols();

		//如果奇异值为0，则本轮应当已经收敛，不再检出（针对重特征值）
		int p = C.cols();
		while (SVDsolver.singularValues()(p - 1, 0) < EIGTOL)
			--p;
		//取对应于最小奇异值的若干向量
		X.noalias() = V * SVDsolver.matrixV().middleCols(p - X.cols(), X.cols());
		com_of_mul += A.rows() * V.cols();
	}

	refineRitz(SparseMatrix<double, RowMajor, __int64>& A, SparseMatrix<double, RowMajor, __int64>& B, int nev, int cgstep, int q, int r, double ratio, int Vsize);
	void compute();
};
#endif
