#ifndef __ITERRITZ_H__
#define __ITERRITZ_H__
#include<LinearEigenSolver.h>

using namespace Eigen;
class IterRitz : public LinearEigenSolver{
public:
	MatrixXd X, V, Lam, P;
	int q, r, cgstep;

	ConjugateGradient<SparseMatrix<double, RowMajor, __int64>, Lower | Upper> linearsolver;

	IterRitz(SparseMatrix<double, RowMajor, __int64>& A, SparseMatrix<double, RowMajor, __int64>& B, int nev, int cgstep, int q, int r);
	void compute();
};
#endif
