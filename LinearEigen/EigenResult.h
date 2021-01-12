#ifndef  __EIGEN_RESULT_H__
#define   __EIGEN_RESULT_H__
#include<Eigen\Sparse>
#include<vector>
using namespace Eigen;
using namespace std;
class EigenResult {
public:
	int N;
	int nev;
	vector<double> eigv;
	vector<SparseMatrix<double>> eigc;
	EigenResult(int N, int nev);
};
#endif