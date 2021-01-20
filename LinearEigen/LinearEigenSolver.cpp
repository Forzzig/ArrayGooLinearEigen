#include<LinearEigenSolver.h>
#include<iostream>

using namespace std;

double LinearEigenSolver::ORTH_TOL = 1e-10;
double LinearEigenSolver::EIGTOL = 1e-3;
int LinearEigenSolver::CHECKNUM = 3;

LinearEigenSolver::LinearEigenSolver(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev) : A(A), B(B), nev(nev), nIter(0) {
	eigenvectors.resize(A.rows(), 0);
}