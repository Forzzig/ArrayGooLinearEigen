#ifndef __RITZ_H__
#define __RITZ_H__
#include<LinearEigenSolver.h>
#include<Eigen/OrderingMethods>
#include <Eigen/PardisoSupport>

#define DIRECT

using namespace Eigen;
class Ritz : public LinearEigenSolver {
public:
	MatrixXd X;
	MatrixXd LAM, V, P;
	int q, r, cgstep;
	int L_nnz;

#ifdef DIRECT
	PardisoLDLT<SparseMatrix<double, RowMajor, __int64>, Upper | Lower> linearsolver;
	//SimplicialLDLT<SparseMatrix<double>, Upper | Lower, COLAMDOrdering<int>> linearsolver;
#endif // DIRECT

	Ritz(SparseMatrix<double, RowMajor, __int64>& A, SparseMatrix<double, RowMajor, __int64>& B, int nev, int cgstep, int q, int r);
	void compute();
};
#endif
