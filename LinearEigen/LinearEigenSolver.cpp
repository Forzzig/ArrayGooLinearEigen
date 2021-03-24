#include<LinearEigenSolver.h>
#include<iomanip>

using namespace std;

double LinearEigenSolver::ORTH_TOL = 1e-10;
double LinearEigenSolver::EIGTOL = 1e-3;
int LinearEigenSolver::CHECKNUM = 1;
ofstream LinearEigenSolver::coutput("coutput.txt", ios::out);

LinearEigenSolver::LinearEigenSolver(SparseMatrix<double, RowMajor>& A, SparseMatrix<double, RowMajor>& B, int nev)
	: A(A),
	B(B), 
	nev(nev), 
	nIter(0),
	eigenvectors(A.rows(), 0),
	com_of_mul(0), 
	start_time(time(NULL)),
	end_time(LLONG_MAX),
	globaltmp(A.rows(), 0) {
	coutput << scientific << setprecision(16);
}