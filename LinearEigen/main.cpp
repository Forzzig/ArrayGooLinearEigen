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

	//��Ҫ��ʱ�������
	//	srand((unsigned)time(NULL));

	//���˿��ľͲ��ÿ�ѧ��������
	//cout << scientific << setprecision(16);
	string matrixName;
	getline(cin, matrixName);
	
	//��A����
	SparseMatrix<double> A;
	if (matrixName.length() > 0)
		A = mtxio::getSparseMatrix("./matrix/" + matrixName + ".mtx");
	else
		A = mtxio::getSparseMatrix("./matrix/bcsstk01.mtx");

	//B���õ�λ��
	SparseMatrix<double> B(A.rows(), A.cols());
	B.reserve(A.rows());
	for (int i = 0; i < A.rows(); ++i)
		B.insert(i, i) = 1;
	cout << "���������" << A.rows() << endl;
	cout << "����Ԫ����" << A.nonZeros() << endl;
	
	/*cout << "A-------------------------" << endl << A << endl;
	cout << "B-------------------------" << endl << B << endl;*/
	system("pause");

	cout << "LOBPCG_I��ʼ���..." << endl;
	//(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep)
	LOBPCG_I LP1(A, B, 20, 40);
	LP1.compute();
	
	
	///*cout << "ԭʼGCG��ʼ���..." << endl;
	//GCG_sv gsv(A, B, 10, 40, 10);
	//gsv.compute();*/
	
	
	//cout << "LOBPCG��ʼ���..." << endl;
	//LOBPCG_solver LOBPCG(A, B, 20, 40);
	//LOBPCG.compute();
	
	
	cout << "����Ritz��ʼ���..." << endl;
	//(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int q, int r) 
	IterRitz iritz(A, B, 20, 20, 20, 3);
	iritz.compute();
	
	
	//cout << "JD��ʼ���..." << endl;
	////(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int restart, int batch, int gmres_size)
	//JD jd(A, B, 20, 100, 10, 5, 10);
	///*jd.compute();*/


	cout << "BJD��ʼ���..." << endl;
	//(SparseMatrix<double> & A, SparseMatrix<double> & B, int nev, int cgstep, int restart, int batch, int gmres_size)
	BJD bjd(A, B, 10, 100, 20, 5, 20);
	//(SparseMatrix<double>& A, SparseMatrix<double>& B, Derived_rhs& b, Derived_ss& U, Derived_sol& X, Derived_eval& lam, int m)
	//bjd.compute();

	//system("pause");
	system("cls");


	for (int i = 0; i < LP1.eigenvalues.size(); ++i) {
		cout << "��" << i + 1 << "������ֵ��" << LP1.eigenvalues[i] << endl;
		//cout << "��" << i + 1 << "������������" << LP1.eigenvectors.col(i).transpose() << endl;
		cout << (A * LP1.eigenvectors.col(i) - LP1.eigenvalues[i] * B * LP1.eigenvectors.col(i)).norm() / (A * LP1.eigenvectors.col(i)).norm() << endl;
	}
	cout << "LOBPCG_I��������" << LP1.nIter << endl;
	cout << "LOBPCG_I�˷�����" << LP1.com_of_mul << endl;


	////for (int i = 0; i < gsv.eigenvalues.size(); ++i) {
	////	cout << "��" << i + 1 << "������ֵ��" << gsv.eigenvalues[i] << endl;
	////	//cout << "��" << i + 1 << "������������" << gcg.eigenvectors.col(i).transpose() << endl;
	////	cout << (A * gsv.eigenvectors.col(i) - gsv.eigenvalues[i] * B * gsv.eigenvectors.col(i)).norm() / (A * gsv.eigenvectors.col(i)).norm() << endl;
	////}
	////cout << "ԭʼGCG��������" << gsv.nIter << endl;


	//for (int i = 0; i < LOBPCG.eigenvalues.size(); ++i) {
	//	cout << "��" << i + 1 << "������ֵ��" << LOBPCG.eigenvalues[i] << endl;
	//	//cout << "��" << i + 1 << "������������" << LOBPCG.eigenvectors.col(i).transpose() << endl;
	//	cout << (A * LOBPCG.eigenvectors.col(i) - LOBPCG.eigenvalues[i] * B * LOBPCG.eigenvectors.col(i)).norm() / (A * LOBPCG.eigenvectors.col(i)).norm() << endl;
	//}
	//cout << "LOBPCG��������" << LOBPCG.nIter << endl;


	for (int i = 0; i < iritz.eigenvalues.size(); ++i) {
		cout << "��" << i + 1 << "������ֵ��" << iritz.eigenvalues[i] << endl;
		//cout << "��" << i + 1 << "������������" << iritz.eigenvectors.col(i).transpose() << endl;
		cout << (A * iritz.eigenvectors.col(i) - iritz.eigenvalues[i] * B * iritz.eigenvectors.col(i)).norm() / (A * iritz.eigenvectors.col(i)).norm() << endl;
	}
	cout << "IterRitz��������" << iritz.nIter << endl;
	cout << "IterRitz�˷�����" << iritz.com_of_mul << endl;


	//for (int i = 0; i < jd.eigenvalues.size(); ++i) {
	//	cout << "��" << i + 1 << "������ֵ��" << jd.eigenvalues[i] << endl;
	//	//cout << "��" << i + 1 << "������������" << LOBPCG.eigenvectors.col(i).transpose() << endl;
	//	cout << (A * jd.eigenvectors.col(i) - jd.eigenvalues[i] * B * jd.eigenvectors.col(i)).norm() / (A * jd.eigenvectors.col(i)).norm() << endl;
	//}
	//cout << "JD��������" << jd.nIter << endl;


	for (int i = 0; i < bjd.eigenvalues.size(); ++i) {
		cout << "��" << i + 1 << "������ֵ��" << bjd.eigenvalues[i] << endl;
		//cout << "��" << i + 1 << "������������" << LOBPCG.eigenvectors.col(i).transpose() << endl;
		cout << (A * bjd.eigenvectors.col(i) - bjd.eigenvalues[i] * B * bjd.eigenvectors.col(i)).norm() / (A * bjd.eigenvectors.col(i)).norm() << endl;
	}
	cout << "BJD��������" << bjd.nIter << endl;
}