#include<iostream>
#include<Eigen\Sparse>
#include<mtxio.h>
#include<EigenResult.h>
#include<LOBPCG_I.h>
#include<GCG_sv.h>
#include<LOBPCG_solver.h>
#include<IterRitz.h>
#include<JD.h>
#include<ctime>
#include<cstdlib>

using namespace std;
using namespace Eigen;

void RR(Block<Map<MatrixXd>>& V) {
	V.col(0)[0] = 1000;
}
void RR(Block<MatrixXd>& V) {
	V.col(0)[0] = 2000;
}

int main() {
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
	SparseMatrix<double> A = mtxio::getSparseMatrix("./matrix/bcsstk17.mtx");
	SparseMatrix<double> B(A.rows(), A.cols());
	B.reserve(A.rows());
	for (int i = 0; i < A.rows(); ++i)
		B.insert(i, i) = 1;
	cout << "矩阵阶数：" << A.rows() << endl;
	cout << "LOBPCG_I开始求解..." << endl;
	LOBPCG_I LP1(A, B, 20, 40);
	LP1.compute();
	///*cout << "原始GCG开始求解..." << endl;
	//GCG_sv gsv(A, B, 10, 40, 10);
	//gsv.compute();*/
	cout << "LOBPCG开始求解..." << endl;
	LOBPCG_solver LOBPCG(A, B, 20, 40);
	//LOBPCG.compute();
	//cout << "迭代Ritz开始求解..." << endl;
	//IterRitz iritz(A, B, 20, 20, 20, 3);
	//iritz.compute();
	cout << "JD开始求解..." << endl;
	JD jd(A, B, 20, 5, 8, 20);
	jd.compute();
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
	//for (int i = 0; i < iritz.eigenvalues.size(); ++i) {
	//	cout << "第" << i + 1 << "个特征值：" << iritz.eigenvalues[i] << endl;
	//	//cout << "第" << i + 1 << "个特征向量：" << LOBPCG.eigenvectors.col(i).transpose() << endl;
	//	cout << (A * iritz.eigenvectors.col(i) - iritz.eigenvalues[i] * B * iritz.eigenvectors.col(i)).norm() / (A * iritz.eigenvectors.col(i)).norm() << endl;
	//}
	//cout << "IterRitz迭代次数" << iritz.nIter << endl;
	for (int i = 0; i < jd.eigenvalues.size(); ++i) {
		cout << "第" << i + 1 << "个特征值：" << jd.eigenvalues[i] << endl;
		//cout << "第" << i + 1 << "个特征向量：" << LOBPCG.eigenvectors.col(i).transpose() << endl;
		cout << (A * jd.eigenvectors.col(i) - jd.eigenvalues[i] * B * jd.eigenvectors.col(i)).norm() / (A * jd.eigenvectors.col(i)).norm() << endl;
	}
	cout << "JD迭代次数" << jd.nIter << endl;
}