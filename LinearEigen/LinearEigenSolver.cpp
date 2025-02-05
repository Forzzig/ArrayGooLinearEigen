#include<LinearEigenSolver.h>
#include<iomanip>

using namespace std;

double LinearEigenSolver::ORTH_TOL = 1e-10;
double LinearEigenSolver::EIGTOL = 1e-3;
int LinearEigenSolver::CHECKNUM = 1;
fstream LinearEigenSolver::coutput("coutput.txt");

LinearEigenSolver::LinearEigenSolver(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev) 
	: A(A),
	B(B), 
	nev(nev), 
	nIter(0),
	eigenvectors(A.rows(), 0),
	com_of_mul(0), 
	start_time(time(NULL)){

	coutput << scientific << setprecision(16);
}