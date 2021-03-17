#ifndef __ITERRITZ_H__
#define __ITERRITZ_H__
#include<LinearEigenSolver.h>

using namespace Eigen;
class IterRitz : public LinearEigenSolver{
public:
	MatrixXd X;
	MatrixXd X1;
	MatrixXd LAM, P;
	int q, r, cgstep;
	IterRitz(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int q, int r);
	void compute();
};
#endif
