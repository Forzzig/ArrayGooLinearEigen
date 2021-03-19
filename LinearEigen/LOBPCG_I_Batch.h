#ifndef __LOBPCG_I_BATCH_H__
#define __LOBPCG_I_BATCH_H__
#include<LinearEigenSolver.h>

using namespace Eigen;

class LOBPCG_I_Batch : public LinearEigenSolver {
public:
	int cgstep, batch;
	double* storage = NULL;
	Map<MatrixXd> X, P, W;
	MatrixXd H;
	LOBPCG_I_Batch(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int batch);
	void compute();
};
#endif
