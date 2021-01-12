//与论文一致的GCG

#ifndef __GCG_SV_H__
#define __GCG_SV_H__
#include<Eigen\Sparse>
#include<LinearEigenSolver.h>

using namespace Eigen;
class GCG_sv : public LinearEigenSolver {
public:
	MatrixXd X0, X, P, W1, W2, V, eval;
	int batch;
	SparseMatrix<double> LAM;
	int conv_check(SparseMatrix<double>& A, SparseMatrix<double>& B, MatrixXd& eval, MatrixXd& evec, double shift, double tol = 1e-5);
	GCG_sv(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep);
	GCG_sv(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int batch);
	void compute();
};
#endif
