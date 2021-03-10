#include<iostream>
#include<Eigen\Sparse>
#include<mtxio.h>
//#include<EigenResult.h>
#include<LOBPCG_I.h>
//#include<GCG_sv.h>
//#include<LOBPCG_solver.h>
#include<IterRitz.h>
//#include<JD.h>
#include<BJD.h>
#include<ctime>
#include<cstdlib>
#include<iomanip>
#include<fstream>

using namespace std;
using namespace Eigen;

ofstream output("coutput.txt");

void RR(Block<Map<MatrixXd>>& V) {
	V.col(0)[0] = 1000;
}
void RR(Block<MatrixXd>& V) {
	V.col(0)[0] = 2000;
}

int main() {

	//需要的时候开随机化
	//	srand((unsigned)time(NULL));

	//给人看的就不用科学计数法了
	//cout << scientific << setprecision(16);
	string matrixName;
	getline(cin, matrixName);
	
	//读A矩阵
	SparseMatrix<double> A;
	if (matrixName.length() > 0)
		A = mtxio::getSparseMatrix("./matrix/" + matrixName + ".mtx");
	else
		A = mtxio::getSparseMatrix("./matrix/bcsstk01.mtx");

	//B先用单位阵
	SparseMatrix<double> B(A.rows(), A.cols());
	B.reserve(A.rows());
	for (int i = 0; i < A.rows(); ++i)
		B.insert(i, i) = 1;
	cout << "矩阵阶数：" << A.rows() << endl;
	cout << "非零元数：" << A.nonZeros() << endl;
	
	/*cout << "A-------------------------" << endl << A << endl;
	cout << "B-------------------------" << endl << B << endl;*/
	system("pause");

	cout << "LOBPCG_I开始求解..." << endl;
	//(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep)
	LOBPCG_I LP1(A, B, 20, 40);
	LP1.compute();
	
	
	///*cout << "原始GCG开始求解..." << endl;
	//GCG_sv gsv(A, B, 10, 40, 10);
	//gsv.compute();*/
	
	
	//cout << "LOBPCG开始求解..." << endl;
	//LOBPCG_solver LOBPCG(A, B, 20, 40);
	//LOBPCG.compute();
	
	
	cout << "迭代Ritz开始求解..." << endl;
	//(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int q, int r) 
	IterRitz iritz(A, B, 20, 20, 20, 3);
	iritz.compute();
	
	
	//cout << "JD开始求解..." << endl;
	////(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int restart, int batch, int gmres_size)
	//JD jd(A, B, 20, 100, 10, 5, 10);
	///*jd.compute();*/


	cout << "BJD开始求解..." << endl;
	//(SparseMatrix<double> & A, SparseMatrix<double> & B, int nev, int cgstep, int restart, int batch, int gmres_size)
	BJD bjd(A, B, 10, 100, 20, 5, 20);
	//(SparseMatrix<double>& A, SparseMatrix<double>& B, Derived_rhs& b, Derived_ss& U, Derived_sol& X, Derived_eval& lam, int m)
	//bjd.compute();

	//system("pause");
	system("cls");


	for (int i = 0; i < LP1.eigenvalues.size(); ++i) {
		cout << "第" << i + 1 << "个特征值：" << LP1.eigenvalues[i] << endl;
		//cout << "第" << i + 1 << "个特征向量：" << LP1.eigenvectors.col(i).transpose() << endl;
		cout << (A * LP1.eigenvectors.col(i) - LP1.eigenvalues[i] * B * LP1.eigenvectors.col(i)).norm() / (A * LP1.eigenvectors.col(i)).norm() << endl;
	}
	cout << "LOBPCG_I迭代次数" << LP1.nIter << endl;
	cout << "LOBPCG_I乘法次数" << LP1.com_of_mul << endl;


	////for (int i = 0; i < gsv.eigenvalues.size(); ++i) {
	////	cout << "第" << i + 1 << "个特征值：" << gsv.eigenvalues[i] << endl;
	////	//cout << "第" << i + 1 << "个特征向量：" << gcg.eigenvectors.col(i).transpose() << endl;
	////	cout << (A * gsv.eigenvectors.col(i) - gsv.eigenvalues[i] * B * gsv.eigenvectors.col(i)).norm() / (A * gsv.eigenvectors.col(i)).norm() << endl;
	////}
	////cout << "原始GCG迭代次数" << gsv.nIter << endl;


	//for (int i = 0; i < LOBPCG.eigenvalues.size(); ++i) {
	//	cout << "第" << i + 1 << "个特征值：" << LOBPCG.eigenvalues[i] << endl;
	//	//cout << "第" << i + 1 << "个特征向量：" << LOBPCG.eigenvectors.col(i).transpose() << endl;
	//	cout << (A * LOBPCG.eigenvectors.col(i) - LOBPCG.eigenvalues[i] * B * LOBPCG.eigenvectors.col(i)).norm() / (A * LOBPCG.eigenvectors.col(i)).norm() << endl;
	//}
	//cout << "LOBPCG迭代次数" << LOBPCG.nIter << endl;


	for (int i = 0; i < iritz.eigenvalues.size(); ++i) {
		cout << "第" << i + 1 << "个特征值：" << iritz.eigenvalues[i] << endl;
		//cout << "第" << i + 1 << "个特征向量：" << iritz.eigenvectors.col(i).transpose() << endl;
		cout << (A * iritz.eigenvectors.col(i) - iritz.eigenvalues[i] * B * iritz.eigenvectors.col(i)).norm() / (A * iritz.eigenvectors.col(i)).norm() << endl;
	}
	cout << "IterRitz迭代次数" << iritz.nIter << endl;
	cout << "IterRitz乘法次数" << iritz.com_of_mul << endl;


	//for (int i = 0; i < jd.eigenvalues.size(); ++i) {
	//	cout << "第" << i + 1 << "个特征值：" << jd.eigenvalues[i] << endl;
	//	//cout << "第" << i + 1 << "个特征向量：" << LOBPCG.eigenvectors.col(i).transpose() << endl;
	//	cout << (A * jd.eigenvectors.col(i) - jd.eigenvalues[i] * B * jd.eigenvectors.col(i)).norm() / (A * jd.eigenvectors.col(i)).norm() << endl;
	//}
	//cout << "JD迭代次数" << jd.nIter << endl;


	for (int i = 0; i < bjd.eigenvalues.size(); ++i) {
		cout << "第" << i + 1 << "个特征值：" << bjd.eigenvalues[i] << endl;
		//cout << "第" << i + 1 << "个特征向量：" << LOBPCG.eigenvectors.col(i).transpose() << endl;
		cout << (A * bjd.eigenvectors.col(i) - bjd.eigenvalues[i] * B * bjd.eigenvectors.col(i)).norm() / (A * bjd.eigenvectors.col(i)).norm() << endl;
	}
	cout << "BJD迭代次数" << bjd.nIter << endl;
}