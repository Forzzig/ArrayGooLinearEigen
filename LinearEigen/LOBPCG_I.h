#ifndef __LOBPCG_I_H__
#define __LOBPCG_I_H__
#include<Eigen\Sparse>
#include<Eigen\Dense>
#include<EigenResult.h>
#include<iostream>
#include<LinearEigenSolver.h>

using namespace Eigen;
using namespace std;

class LOBPCG_I : public LinearEigenSolver{
public:
	double* storage = NULL;
	Map<MatrixXd> X, P, W;
	MatrixXd LAM;
	LOBPCG_I(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep);
	void compute();
};
#endif
