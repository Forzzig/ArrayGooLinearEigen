#ifndef __RITZ_H__
#define __RITZ_H__
#include<LinearEigenSolver.h>

#define DIRECT

using namespace Eigen;
class Ritz : public LinearEigenSolver {
public:
	MatrixXd X;
	MatrixXd LAM, V, P;
	int q, r, cgstep;

#ifdef DIRECT
	SimplicialLDLT<SparseMatrix<double>> linearsolver;
	SparseMatrix<double> L;
#endif // DIRECT

	Ritz(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int q, int r);
	void compute();
};
#endif
