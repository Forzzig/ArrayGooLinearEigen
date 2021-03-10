#include<LinearEigenSolver.h>
#include<iostream>

using namespace std;

double LinearEigenSolver::ORTH_TOL = 1e-10;
double LinearEigenSolver::EIGTOL = 1e-3;
int LinearEigenSolver::CHECKNUM = 3;
fstream LinearEigenSolver::coutput("coutput.txt");

LinearEigenSolver::LinearEigenSolver(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev) : A(A), B(B), nev(nev), nIter(0) {
	com_of_mul = 0;
	eigenvectors.resize(A.rows(), 0);
	coutput << scientific << setprecision(16);
}