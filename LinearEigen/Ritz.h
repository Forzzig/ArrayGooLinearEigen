#ifndef __RITZ_H__
#define __RITZ_H__
#include<LinearEigenSolver.h>
#include<Eigen/OrderingMethods>

#define DIRECT

using namespace Eigen;
class Ritz : public LinearEigenSolver {
public:
	MatrixXd X;
	MatrixXd LAM, V, P;
	int q, r, cgstep;
	int L_nnz;

#ifdef DIRECT
	SimplicialLDLT<SparseMatrix<double>, Upper | Lower, COLAMDOrdering<int>> linearsolver;
#endif // DIRECT

	Ritz(SparseMatrix<double, RowMajor>& A, SparseMatrix<double, RowMajor>& B, int nev, int cgstep, int q, int r);
	void compute();
};
#endif
