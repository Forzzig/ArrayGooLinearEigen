#include<iostream>
#include<Eigen\Sparse>
#include<mtxio.h>
//#include<EigenResult.h>
#include<LOBPCG_I.h>
//#include<GCG_sv.h>
//#include<LOBPCG_solver.h>
#include<IterRitz.h>
#include<JD.h>
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


	MatrixXd H(10, 10);
	/*H << 
		3.3453224658497451e+06, 1.1922410162192157e+06, -9.3115541279504122e+05, 1.5977475053119021e+06, -6.5536939867209392e+06, 4.3396825729101095e+06, 8.8708148129458434e+05, 1.2925665797375501e+06, 3.0255102667339379e+06, 1.7955467404403989e+06,
	1.1922410162192159e+06, 4.1021083509421661e+07, -2.1051516561355989e+07, 1.1789107831681101e+06, 1.7241808929501161e+07, 8.8708148129458388e+05, 8.6937993718033329e+07, 5.5181310907210946e+07, 2.5920424694375981e+07, 1.0288264800505089e+07,
	-9.3115541279504076e+05, -2.1051516561355993e+07, 3.5731593339602321e+07, 1.5159271641547222e+07, -4.7348967705906890e+07, 1.2925665797375496e+06, 5.5181310907210991e+07, 2.1963942258537278e+08, 6.5466190446795106e+07, -1.1306628037057877e+07,
	1.5977475053119017e+06, 1.1789107831681068e+06, 1.5159271641547238e+07, 1.0401289629788743e+08, -3.0711817118526205e+07, 3.0255102667339388e+06, 2.5920424694375996e+07, 6.5466190446795076e+07, 1.2768797579122224e+08, -2.7908764140825965e+07,
	-6.5536939867209392e+06, 1.7241808929501165e+07, -4.7348967705906905e+07, -3.0711817118526205e+07, 5.5862414584199023e+08, 1.7955467404403996e+06, 1.0288264800505091e+07, -1.1306628037057878e+07, -2.7908764140825965e+07, 4.1405671156207681e+08,
	4.3396825729101095e+06, 8.8708148129458388e+05, 1.2925665797375496e+06, 3.0255102667339388e+06, 1.7955467404403996e+06, 4.3396825729101095e+06, 8.8708148129458388e+05, 1.2925665797375496e+06, 3.0255102667339388e+06, 1.7955467404403996e+06,
	8.8708148129458434e+05, 8.6937993718033329e+07, 5.5181310907210991e+07, 2.5920424694375996e+07, 1.0288264800505091e+07, 8.8708148129458434e+05, 8.6937993718033329e+07, 5.5181310907210991e+07, 2.5920424694375996e+07, 1.0288264800505091e+07,
	1.2925665797375501e+06, 5.5181310907210946e+07, 2.1963942258537278e+08, 6.5466190446795076e+07, -1.1306628037057878e+07, 1.2925665797375501e+06, 5.5181310907210946e+07, 2.1963942258537278e+08, 6.5466190446795076e+07, -1.1306628037057878e+07,
	3.0255102667339379e+06, 2.5920424694375981e+07, 6.5466190446795106e+07, 1.2768797579122224e+08, -2.7908764140825965e+07, 3.0255102667339379e+06, 2.5920424694375981e+07, 6.5466190446795106e+07, 1.2768797579122224e+08, -2.7908764140825965e+07,
	1.7955467404403989e+06, 1.0288264800505089e+07, -1.1306628037057877e+07, -2.7908764140825965e+07, 4.1405671156207681e+08, 1.7955467404403989e+06, 1.0288264800505089e+07, -1.1306628037057877e+07, -2.7908764140825965e+07, 4.1405671156207681e+08;
	SelfAdjointEigenSolver<MatrixXd> eigensolver;
	eigensolver.compute(H);
	cout << eigensolver.eigenvalues() << endl;
	system("pause");*/
}
	//	srand((unsigned)time(NULL));
	/*SparseMatrix<double> tmpA;
	MatrixXd tmpB;
	MatrixXd tmpC;
	tmpA.resize(3, 4);
	tmpA.reserve(12);
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 4; ++j) {
			tmpA.insert(i, j) = i * (j + 1) + 3.0 * j / (i + 1);
		}
	}
	tmpB.resize(3, 3);
	tmpB << 9, 3, 2,
		21, 8, 2,
		4, 5, 7;
	tmpC.resize(4, 4);
	tmpC << 2, 23, 4, 1,
		4, 6, 2, 67,
		3, 87, 23, 45,
		6, 12, 76, 3;
	cout << tmpA << endl << tmpB << endl << tmpC << endl;
	double a[100];
	for (int i = 0; i < 100; ++i)
		a[i] = 3 * i - 2.0 / (i + 1);
	Map<MatrixXd> tmpD(a, 3, 4);
	cout << tmpD << endl;
	cout << tmpD * tmpC << endl;
	Map<MatrixXd> tmpE(a + 12, 4, 3);
	cout << tmpE << endl;
	tmpE *= tmpB;
	cout << tmpE << endl;
	for (int i = 12; i < 24; ++i)
		cout << a[i] << " " << endl;
	tmpE.col(0) = tmpC.col(0);
	cout << tmpE << endl;
	cout << tmpE.block(2, 1, 2, 2) << endl;
	tmpE.block(1, 0, 3, 3) = tmpB;
	Map<MatrixXd> vv(&tmpB(1, 1), 2, 2);
	cout << tmpE << endl;
	RR(tmpE.block(1, 0, 3, 3));
	cout << tmpE << endl;
	cout << tmpB << endl;
	RR(vv.block(0, 0, 2, 2));
	cout << tmpB << endl;
	RR(tmpB.block(0, 0, 2, 2));
	cout << tmpB << endl;
	tmpD = tmpA;
	cout << tmpD << endl;
	for (int i = 0; i < 12; ++i)
		cout << a[i] << " " << endl;*/

	cout << scientific << setprecision(16);
	SparseMatrix<double> A = mtxio::getSparseMatrix("./matrix/bcsstk01.mtx");
	SparseMatrix<double> B(A.rows(), A.cols());
	B.reserve(A.rows());
	for (int i = 0; i < A.rows(); ++i)
		B.insert(i, i) = 1;
	cout << "矩阵阶数：" << A.rows() << endl;
	
	/*cout << "A-------------------------" << endl << A << endl;
	cout << "B-------------------------" << endl << B << endl;
	system("pause");*/

	cout << "LOBPCG_I开始求解..." << endl;
	LOBPCG_I LP1(A, B, 20, 40);
	//LP1.compute();
	
	
	///*cout << "原始GCG开始求解..." << endl;
	//GCG_sv gsv(A, B, 10, 40, 10);
	//gsv.compute();*/
	
	
	//cout << "LOBPCG开始求解..." << endl;
	//LOBPCG_solver LOBPCG(A, B, 20, 40);
	//LOBPCG.compute();
	
	
	cout << "迭代Ritz开始求解..." << endl;
	IterRitz iritz(A, B, 20, 10, 20, 3);
	//iritz.compute();
	
	
	cout << "JD开始求解..." << endl;
	//(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int restart, int batch, int gmres_size)
	JD jd(A, B, 20, 100, 10, 5, 10);
	/*jd.compute();*/


	cout << "BJD开始求解..." << endl;
	//(SparseMatrix<double> & A, SparseMatrix<double> & B, int nev, int cgstep, int restart, int batch, int gmres_size)
	BJD bjd(A, B, 10, 20, 3, 5, 5);
	//(SparseMatrix<double>& A, SparseMatrix<double>& B, Derived_rhs& b, Derived_ss& U, Derived_sol& X, Derived_eval& lam, int m)
	/*MatrixXd b = MatrixXd::Random(A.rows(), 1);
	MatrixXd U;
	MatrixXd X = MatrixXd::Zero(A.rows(), 1);
	MatrixXd lam;
	lam.resize(1, 1);
	lam(0, 0) = 0;
	U.resize(A.rows(), 0);
	cout << "b------------------------" << endl << b << endl;

	bjd.L_GMRES(A, B, b, U, X, lam, 5);
	cout << "X------------------------" << endl << X << endl;
	cout << "A * X------------------------" << endl << A * X << endl;
		system("pause");*/
	bjd.compute();


	system("cls");


	for (int i = 0; i < LP1.eigenvalues.size(); ++i) {
		cout << "第" << i + 1 << "个特征值：" << LP1.eigenvalues[i] << endl;
		//cout << "第" << i + 1 << "个特征向量：" << LP1.eigenvectors.col(i).transpose() << endl;
		cout << (A * LP1.eigenvectors.col(i) - LP1.eigenvalues[i] * B * LP1.eigenvectors.col(i)).norm() / (A * LP1.eigenvectors.col(i)).norm() << endl;
	}
	cout << "LOBPCG_I迭代次数" << LP1.nIter << endl;


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
		//cout << "第" << i + 1 << "个特征向量：" << LOBPCG.eigenvectors.col(i).transpose() << endl;
		cout << (A * iritz.eigenvectors.col(i) - iritz.eigenvalues[i] * B * iritz.eigenvectors.col(i)).norm() / (A * iritz.eigenvectors.col(i)).norm() << endl;
	}
	cout << "IterRitz迭代次数" << iritz.nIter << endl;


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