#include<iostream>
#include<Eigen\Sparse>
#include<mtxio.h>
//#include<EigenResult.h>
#include<LOBPCG_I.h>
#include<LOBPCG_II.h>
//#include<GCG_sv.h>
//#include<LOBPCG_solver.h>
#include<IterRitz.h>
#include<Ritz.h>
//#include<JD.h>
#include<BJD.h>
#include<ctime>
#include<cstdlib>
#include<iomanip>
#include<fstream>
#include<string>

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

	//���־���
	string matrices[1000] =
	{	"bcsstk01",
		"bcsstk02",
		"bcsstk05", 
		"bcsstk07",
		"bcsstk08",
		"bcsstk09",
		"bcsstk10",
		"bcsstk12",
		"bcsstk13",
		"bcsstk15",
		"bcsstk16",
		"bcsstk17",
		"bcsstk18",
		"bcsstk19",
		"bwm2000",
		"fidapm29",
		"s3rmt3m3",
		"1138_bus"
	};

	ofstream result;
	int n_matrices = 0;
	while (matrices[n_matrices].length() != 0) {

		//��Ҫ��ʱ�������
		//	srand((unsigned)time(NULL));

		//���˿��ľͲ��ÿ�ѧ��������
		//cout << scientific << setprecision(16);

		//��A����
		string matrixName = matrices[n_matrices];
		SparseMatrix<double> A;
		if (matrixName.length() > 0)
			A = mtxio::getSparseMatrix("./matrix/" + matrixName + ".mtx");
		else
			A = mtxio::getSparseMatrix("./matrix/bcsstk01.mtx");

		//TODO B���õ�λ��
		SparseMatrix<double> B(A.rows(), A.cols());
		B.reserve(A.rows());
		for (int i = 0; i < A.rows(); ++i)
			B.insert(i, i) = 1;
		

		/*cout << "A-------------------------" << endl << A << endl;
		cout << "B-------------------------" << endl << B << endl;*/
		//system("pause");

		cout << "�������" << matrixName << "....................." << endl;

		cout << "��" << matrixName << "ʹ��LOBPCG_I....................." << endl;
		
		result.open(matrixName + "-LOBPCG_I.txt");
		result << "���������" << A.rows() << endl;
		result << "����Ԫ����" << A.nonZeros() << endl << endl;
		result << matrixName + "��LOBPCG_I��ʼ���........................................." << endl;
		for (int nev = 10; nev <= 50; nev += 10) {
			if (A.rows() / nev < 3)
				break;
			for (int cgstep = 10; cgstep <= 50; cgstep += 10) {
				if (A.rows() / cgstep < 2)
					break;
				//(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep)
				result << "LOBPCG_Iִ�в�����" << endl << "����ֵ��" << nev << "�������CG��������" << cgstep << "��" << endl;
				cout << "LOBPCG_Iִ�в�����" << endl << "����ֵ��" << nev << "�������CG��������" << cgstep << "��" << endl;
				LOBPCG_I LP1(A, B, nev, cgstep);
				LP1.compute();

				for (int i = 0; i < LP1.eigenvalues.size(); ++i) {
					result << "��" << i + 1 << "������ֵ��" << LP1.eigenvalues[i] << "��";
					//cout << "��" << i + 1 << "������������" << LP1.eigenvectors.col(i).transpose() << endl;
					result << "�����" << (A * LP1.eigenvectors.col(i) - LP1.eigenvalues[i] * B * LP1.eigenvectors.col(i)).norm() / (A * LP1.eigenvectors.col(i)).norm() << endl;
				}
				result << "LOBPCG_I����������" << LP1.nIter << endl;
				result << "LOBPCG_I�˷�������" << LP1.com_of_mul << endl << endl;
			}
		}
		cout << "��" << matrixName << "ʹ��LOBPCG_I������" << endl;
		result.close();

		result.open(matrixName + "-LOBPCG_II.txt");
		result << "���������" << A.rows() << endl;
		result << "����Ԫ����" << A.nonZeros() << endl << endl;
		cout << "��" << matrixName << "ʹ��LOBPCG_II....................." << endl;
		result << matrixName + "��LOBPCG_II��ʼ���........................................." << endl;
		for (int nev = 10; nev <= 50; nev += 10) {
			if (A.rows() / nev < 3)
				break;
			for (int cgstep = 10; cgstep <= 50; cgstep += 10) {
				if (A.rows() / cgstep < 2)
					break;
				//(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep)
				result << "LOBPCG_IIִ�в�����" << endl << "����ֵ��" << nev << "�������CG��������" << cgstep << "��" << endl;
				cout << "LOBPCG_IIִ�в�����" << endl << "����ֵ��" << nev << "�������CG��������" << cgstep << "��" << endl;
				LOBPCG_II LP2(A, B, nev, cgstep);
				LP2.compute();

				for (int i = 0; i < LP2.eigenvalues.size(); ++i) {
					result << "��" << i + 1 << "������ֵ��" << LP2.eigenvalues[i] << "��";
					//cout << "��" << i + 1 << "������������" << LP1.eigenvectors.col(i).transpose() << endl;
					result << "�����" << (A * LP2.eigenvectors.col(i) - LP2.eigenvalues[i] * B * LP2.eigenvectors.col(i)).norm() / (A * LP2.eigenvectors.col(i)).norm() << endl;
				}
				result << "LOBPCG_II����������" << LP2.nIter << endl;
				result << "LOBPCG_II�˷�������" << LP2.com_of_mul << endl << endl;
			}
		}
		cout << "��" << matrixName << "ʹ��LOBPCG_II������" << endl;
		result.close();

		result.open(matrixName + "-BJD.txt");
		result << "���������" << A.rows() << endl;
		result << "����Ԫ����" << A.nonZeros() << endl << endl;
		cout << "��" << matrixName << "ʹ�ÿ�J-D....................." << endl;
		result << matrixName + "����J-D��ʼ���........................................." << endl;
		for (int batch = 5; batch <= 20; batch += 5) {
			for (int nev = 10; nev <= 50; nev += 10) {
				if (nev < batch)
					continue;
				if (A.rows() / nev < 3)
					break;
				for (int restart = 5; restart <= 20; restart += 5) {
					if (A.rows() < batch * restart)
						break;
					for (int gmres_size = 5; gmres_size <= 20; gmres_size += 5) {
						for (int gmres_restart = 2; gmres_restart <= 10; gmres_restart += 2) {
							//(SparseMatrix<double> & A, SparseMatrix<double> & B, int nev, int cgstep, int restart, int batch, int gmres_size)
							result << "��J-Dִ�в�����" << endl << "����ֵ��" << nev << "��������������" << restart << "��batch��С��" << batch << endl << 
								"    GMRES�ܵ�����������չ�ռ��������������" << gmres_size * gmres_restart << "��GMRES��չ�ռ��С��" << gmres_size << endl;
							cout << "��J-Dִ�в�����" << endl << "����ֵ��" << nev << "��������������" << restart << "��batch��С��" << batch << endl <<
								"    GMRES�ܵ�����������չ�ռ��������������" << gmres_size * gmres_restart << "��GMRES��չ�ռ��С��" << gmres_size << endl;
							BJD bjd(A, B, nev, gmres_size * gmres_restart, restart, batch, gmres_size);
							//L_GMRES(SparseMatrix<double>& A, SparseMatrix<double>& B, Derived_rhs& b, Derived_ss& U, Derived_sol& X, Derived_eval& lam, int m)
							bjd.compute();

							for (int i = 0; i < bjd.eigenvalues.size(); ++i) {
								result << "��" << i + 1 << "������ֵ��" << bjd.eigenvalues[i] << "��";
								//cout << "��" << i + 1 << "������������" << LOBPCG.eigenvectors.col(i).transpose() << endl;
								result << "�����" << (A * bjd.eigenvectors.col(i) - bjd.eigenvalues[i] * B * bjd.eigenvectors.col(i)).norm() / (A * bjd.eigenvectors.col(i)).norm() << endl;
							}
							result << "��J-D��������" << bjd.nIter << endl;
							result << "��J-D�˷�����" << bjd.com_of_mul << endl << endl;
						}
					}
				}
			}
		}
		cout << "��" << matrixName << "ʹ�ÿ�J-D������" << endl;
		result.close();

		result.open(matrixName + "-Ritz.txt");
		result << "���������" << A.rows() << endl;
		result << "����Ԫ����" << A.nonZeros() << endl << endl;
		cout << "��" << matrixName << "ʹ�õ���Ritz��....................." << endl;
		result << matrixName + "������Ritz����ʼ���........................................." << endl;
		for (int batch = 5; batch <= 20; batch += 5) {
			for (int nev = 10; nev <= 50; nev += 10) {
				if (nev < batch)
					continue;
				if (A.rows() / nev < 3)
					break;
				for (int r = 3; r <= 10; ++r) {
					if (A.rows() < batch * r)
						break;
					for (int cgstep = 10; cgstep <= 50; cgstep += 10) {
						if (A.rows() / cgstep < 2)
							break;
						//(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int q, int r) 
						result << "����Ritz��ִ�в�����" << endl << "����ֵ��" << nev << "����batch��С��" << batch << "��Ritz������չ������" << r << ",���CG��������" << cgstep << endl;
						cout << "����Ritz��ִ�в�����" << endl << "����ֵ��" << nev << "����batch��С��" << batch << "��Ritz������չ������" << r << ",���CG��������" << cgstep << endl;
						Ritz ritz(A, B, nev, cgstep, batch, r);
						ritz.compute();

						for (int i = 0; i < ritz.eigenvalues.size(); ++i) {
							result << "��" << i + 1 << "������ֵ��" << ritz.eigenvalues[i] << "��";
							//cout << "��" << i + 1 << "������������" << ritz.eigenvectors.col(i).transpose() << endl;
							result << "�����" << (A * ritz.eigenvectors.col(i) - ritz.eigenvalues[i] * B * ritz.eigenvectors.col(i)).norm() / (A * ritz.eigenvectors.col(i)).norm() << endl;
						}
						result << "����Ritz����������" << ritz.nIter << endl;
						result << "����Ritz���˷�����" << ritz.com_of_mul << endl;
					}
				}
			}
		}
		cout << "��" << matrixName << "ʹ�õ���Ritz��������" << endl;
		result.close();

		result.open(matrixName + "-iRitz.txt");
		result << "���������" << A.rows() << endl;
		result << "����Ԫ����" << A.nonZeros() << endl << endl;
		cout << "��" << matrixName << "ʹ�øĽ�Ritz��....................." << endl;
		result << matrixName + "���Ľ�Ritz����ʼ���........................................." << endl;
		for (int batch = 5; batch <= 20; batch += 5) {
			for (int nev = 10; nev <= 50; nev += 10) {
				if (nev < batch)
					continue;
				if (A.rows() / nev < 3)
					break;
				for (int r = 3; r <= 10; ++r) {
					if (A.rows() < batch * r)
						break;
					for (int cgstep = 10; cgstep <= 50; cgstep += 10) {
						if (A.rows() / cgstep < 2)
							break;
						//(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int q, int r) 
						result << "�Ľ�Ritz��ִ�в�����" << endl << "����ֵ��" << nev << "����batch��С��" << batch << "��Ritz������չ������" << r << ",���CG��������" << cgstep << endl;
						cout << "�Ľ�Ritz��ִ�в�����" << endl << "����ֵ��" << nev << "����batch��С��" << batch << "��Ritz������չ������" << r << ",���CG��������" << cgstep << endl;
						IterRitz iritz(A, B, nev, cgstep, batch, r);
						iritz.compute();

						for (int i = 0; i < iritz.eigenvalues.size(); ++i) {
							result << "��" << i + 1 << "������ֵ��" << iritz.eigenvalues[i] << "��";
							//cout << "��" << i + 1 << "������������" << iritz.eigenvectors.col(i).transpose() << endl;
							result << "�����" << (A * iritz.eigenvectors.col(i) - iritz.eigenvalues[i] * B * iritz.eigenvectors.col(i)).norm() / (A * iritz.eigenvectors.col(i)).norm() << endl;
						}
						result << "�Ľ�Ritz����������" << iritz.nIter << endl;
						result << "�Ľ�Ritz���˷�����" << iritz.com_of_mul << endl;
					}
				}
			}
		}
		cout << "��" << matrixName << "ʹ�øĽ�Ritz��������" << endl;
		result.close();
		system("cls");
		//system("pause");

		///*cout << "ԭʼGCG��ʼ���..." << endl;
		//GCG_sv gsv(A, B, 10, 40, 10);
		//gsv.compute();*/

		//cout << "LOBPCG��ʼ���..." << endl;
		//LOBPCG_solver LOBPCG(A, B, 20, 40);
		//LOBPCG.compute();

		//cout << "JD��ʼ���..." << endl;
		////(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int restart, int batch, int gmres_size)
		//JD jd(A, B, 20, 100, 10, 5, 10);
		///*jd.compute();*/


		
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



		//for (int i = 0; i < jd.eigenvalues.size(); ++i) {
		//	cout << "��" << i + 1 << "������ֵ��" << jd.eigenvalues[i] << endl;
		//	//cout << "��" << i + 1 << "������������" << LOBPCG.eigenvectors.col(i).transpose() << endl;
		//	cout << (A * jd.eigenvectors.col(i) - jd.eigenvalues[i] * B * jd.eigenvectors.col(i)).norm() / (A * jd.eigenvectors.col(i)).norm() << endl;
		//}
		//cout << "JD��������" << jd.nIter << endl;

	}
	system("pause");
}