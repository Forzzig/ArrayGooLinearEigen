#ifndef __ITERRITZ_H__
#define __ITERRITZ_H__
#include<LinearEigenSolver.h>

using namespace Eigen;
class IterRitz : public LinearEigenSolver{
public:
	MatrixXd X, V, Lam, P;
	int q, r, cgstep;

	ConjugateGradient<SparseMatrix<double, RowMajor>, Lower | Upper> linearsolver;

	IterRitz(SparseMatrix<double, RowMajor>& A, SparseMatrix<double, RowMajor>& B, int nev, int cgstep, int q, int r);
	void compute();
};
#endif
