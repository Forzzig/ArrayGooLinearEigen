#ifndef __ITERRITZ_H__
#define __ITERRITZ_H__
#include<Eigen\Sparse>
#include<Eigen\Dense>
#include<EigenResult.h>
#include<iostream>
#include<LinearEigenSolver.h>

using namespace Eigen;
class IterRitz : public LinearEigenSolver{
public:
	MatrixXd X;
	MatrixXd X1;
	MatrixXd LAM, V, P;
	int q, r;
	IterRitz(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int q, int r);
	void compute();
};
#endif
