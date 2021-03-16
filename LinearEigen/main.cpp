#include<iostream>
#include<Eigen\Sparse>
#include<mtxio.h>
//#include<EigenResult.h>
#include<LOBPCG_I.h>
#include<LOBPCG_I_Batch.h>
#include<LOBPCG_II.h>
#include<LOBPCG_II_Batch.h>
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

ofstream output;

#define mLOBPCG_I
#define mLOBPCG_II
#define mBJD
#define mRitz
#define miRitz

//���־���
string matrices[1000] =
{ "bcsstk01",
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

int main() {

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

#ifdef mLOBPCG_I
		cout << "��" << matrixName << "ʹ��LOBPCG_I....................." << endl;
		
		result.open(matrixName + "-LOBPCG_I.txt");
		result << "���������" << A.rows() << endl;
		result << "����Ԫ����" << A.nonZeros() << endl << endl;
		result << matrixName + "��LOBPCG_I��ʼ���........................................." << endl;
		output.open(matrixName + "-LOBPCG-I-statistics.txt");
		output << "nev, batch, cgstep, iter, multi" << endl;
		for (int nev = 10; nev <= 50; nev += 10) {
			if (A.rows() / nev < 3)
				break;
			for (int batch = 5; batch <= 20; batch += 5) {
				if (batch > nev)
					break;
				long long best = LLONG_MAX;
				int best_step;
				for (int cgstep = 10; cgstep <= 50; cgstep += 10) {
					if (A.rows() / cgstep < 2)
						break;
					//(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep)
					result << "LOBPCG_Iִ�в�����" << endl << "����ֵ��" << nev << "����batch��СΪ" << batch << "�����CG��������" << cgstep << "��" << endl;
					cout << "LOBPCG_Iִ�в�����" << endl << "����ֵ��" << nev << "����batch��СΪ" << batch << "�����CG��������" << cgstep << "��" << endl;
					LOBPCG_I_Batch LP1(A, B, nev, cgstep, batch);
					LP1.compute();

					for (int i = 0; i < LP1.eigenvalues.size(); ++i) {
						result << "��" << i + 1 << "������ֵ��" << LP1.eigenvalues[i] << "��";
						//cout << "��" << i + 1 << "������������" << LP1.eigenvectors.col(i).transpose() << endl;
						result << "�����" << (A * LP1.eigenvectors.col(i) - LP1.eigenvalues[i] * B * LP1.eigenvectors.col(i)).norm() / (A * LP1.eigenvectors.col(i)).norm() << endl;
					}
					result << "LOBPCG_I����������" << LP1.nIter << endl;
					result << "LOBPCG_I�˷�������" << LP1.com_of_mul << endl << endl;
					if (LP1.com_of_mul < best) {
						best = LP1.com_of_mul;
						best_step = cgstep;
					}
					output << nev << ", " << batch << ", " << cgstep << ", " << LP1.nIter << ", " << LP1.com_of_mul << endl;
				}
				result << "����" << nev << "������������batch��СΪ" << batch << "��������Ҫ" << best << "�γ˷���cgstep�趨Ϊ" << best_step << endl << endl << endl;
			}
		}
		cout << "��" << matrixName << "ʹ��LOBPCG_I������" << endl;
		result.close();
		output.close();
#endif

#ifdef mLOBPCG_II
		result.open(matrixName + "-LOBPCG_II.txt");
		result << "���������" << A.rows() << endl;
		result << "����Ԫ����" << A.nonZeros() << endl << endl;
		cout << "��" << matrixName << "ʹ��LOBPCG_II....................." << endl;
		result << matrixName + "��LOBPCG_II��ʼ���........................................." << endl;
		output.open(matrixName + "-LOBPCG-II-statistics.txt");
		output << "nev, batch, cgstep, iter, multi" << endl;
		for (int nev = 10; nev <= 50; nev += 10) {
			if (A.rows() / nev < 3)
				break;
			for (int batch = 5; batch <= 20; batch += 5) {
				if (batch > nev)
					break;
				long long best = LLONG_MAX;
				int best_step;
				for (int cgstep = 10; cgstep <= 50; cgstep += 10) {
					if (A.rows() / cgstep < 2)
						break;
					//(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep)
					result << "LOBPCG_IIִ�в�����" << endl << "����ֵ��" << nev << "����batch��СΪ" << batch << "�����CG��������" << cgstep << "��" << endl;
					cout << "LOBPCG_IIִ�в�����" << endl << "����ֵ��" << nev << "����batch��СΪ" << batch << "�����CG��������" << cgstep << "��" << endl;
					LOBPCG_II_Batch LP2(A, B, nev, cgstep, batch);
					LP2.compute();

					for (int i = 0; i < LP2.eigenvalues.size(); ++i) {
						result << "��" << i + 1 << "������ֵ��" << LP2.eigenvalues[i] << "��";
						//cout << "��" << i + 1 << "������������" << LP1.eigenvectors.col(i).transpose() << endl;
						result << "�����" << (A * LP2.eigenvectors.col(i) - LP2.eigenvalues[i] * B * LP2.eigenvectors.col(i)).norm() / (A * LP2.eigenvectors.col(i)).norm() << endl;
					}
					result << "LOBPCG_II����������" << LP2.nIter << endl;
					result << "LOBPCG_II�˷�������" << LP2.com_of_mul << endl << endl;
					if (LP2.com_of_mul < best) {
						best = LP2.com_of_mul;
						best_step = cgstep;
					}
					output << nev << ", " << batch << ", " << cgstep << ", " << LP2.nIter << ", " << LP2.com_of_mul << endl;
				}
				result << "����" << nev << "������������batch��СΪ" << batch << "��������Ҫ" << best << "�γ˷���cgstep�趨Ϊ" << best_step << endl << endl << endl;
			}
		}
		cout << "��" << matrixName << "ʹ��LOBPCG_II������" << endl;
		result.close();
		output.close();
#endif
#ifdef mBJD
		result.open(matrixName + "-BJD.txt");
		result << "���������" << A.rows() << endl;
		result << "����Ԫ����" << A.nonZeros() << endl << endl;
		cout << "��" << matrixName << "ʹ�ÿ�J-D....................." << endl;
		result << matrixName + "����J-D��ʼ���........................................." << endl;
		output.open(matrixName + "-BJD-statistics.txt");
		output << "nev, batch, restart, gmres_size, gmres_restart, gmres_step, iter, multi" << endl;
		for (int nev = 10; nev <= 50; nev += 10) {
			if (A.rows() / nev < 3)
				break;
			for (int batch = 5; batch <= 20; batch += 5) {
				if (batch > nev)
					break;

				long long best = LLONG_MAX;
				int best_gmres_size, best_gmres_restart, best_restart;
				for (int restart = 5; restart <= 20; restart += 5) {
					if (A.rows() / (batch * restart) < 2)
						break;
					for (int gmres_size = 5; gmres_size <= 15; gmres_size += 5) {
						if (A.rows() / gmres_size < 2)
							break;
						for (int gmres_restart = 1; gmres_restart <= 3; ++gmres_restart) {
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
							if (bjd.com_of_mul < best) {
								best = bjd.com_of_mul;
								best_gmres_size = gmres_size;
								best_gmres_restart = gmres_restart;
								best_restart = restart;
							}
							output << nev << ", " << batch << ", " << restart << ", " << gmres_size << ", " << gmres_restart << ", " 
								<< gmres_size * gmres_restart << ", " << bjd.nIter << ", " << bjd.com_of_mul << endl;
						}
					}
				}
				result << "����" << nev << "������������batchΪ" << batch << "��������Ҫ" << best
					<< "�γ˷����趨Ϊ����" << best_restart << "��������gmres�趨��չ�ռ��С" << best_gmres_size << "���ܵ�������" << best_gmres_size * best_gmres_restart << endl << endl << endl;
			}
		}
		cout << "��" << matrixName << "ʹ�ÿ�J-D������" << endl;
		result.close();
		output.close();
#endif
#ifdef mRitz
		result.open(matrixName + "-Ritz.txt");
		result << "���������" << A.rows() << endl;
		result << "����Ԫ����" << A.nonZeros() << endl << endl;
		cout << "��" << matrixName << "ʹ�õ���Ritz��....................." << endl;
		result << matrixName + "������Ritz����ʼ���........................................." << endl;
		output.open(matrixName + "-Ritz-statistics.txt");
		output << "nev, batch, r, cgstep, iter, multi" << endl;
		for (int nev = 10; nev <= 50; nev += 10) {
			if (A.rows() / nev < 3)
				break;
			for (int batch = 5; batch <= 20; batch += 5) {
				if (batch > nev)
					break;

				long long best = LLONG_MAX;
				int best_r, best_step;
				for (int r = 3; r <= 10; ++r) {
					if (A.rows() / (batch * r) < 2)
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
						result << "����Ritz���˷�����" << ritz.com_of_mul << endl << endl;
						if (ritz.com_of_mul < best) {
							best = ritz.com_of_mul;
							best_r = r;
							best_step = cgstep;
						}
						output << nev << ", " << batch << ", " << r << ", " << cgstep << ", " << ritz.nIter << ", " << ritz.com_of_mul << endl;
					}
				}
				result << "����" << nev << "������������batchΪ" << batch << "��������Ҫ" << best
					<< "�γ˷����趨�������Ϊ" << best_r << "��cgstep�趨Ϊ" << best_step << endl << endl << endl;
			}
		}
		cout << "��" << matrixName << "ʹ�õ���Ritz��������" << endl;
		result.close();
		output.close();
#endif
#ifdef miRitz
		result.open(matrixName + "-iRitz.txt");
		result << "���������" << A.rows() << endl;
		result << "����Ԫ����" << A.nonZeros() << endl << endl;
		cout << "��" << matrixName << "ʹ�øĽ�Ritz��....................." << endl;
		result << matrixName + "���Ľ�Ritz����ʼ���........................................." << endl;
		output.open(matrixName + "-IterRitz-statistics.txt");
		output << "nev, batch, r, cgstep, iter, multi" << endl;
		for (int nev = 10; nev <= 50; nev += 10) {
			for (int batch = 5; batch <= 20; batch += 5) {
				if (nev < batch)
					break;
				if (A.rows() / nev < 3)
					break;

				long long best = LLONG_MAX;
				int best_r, best_step;
				for (int r = 3; r <= 10; ++r) {
					if (A.rows() / (batch * r)< 2)
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
						result << "�Ľ�Ritz���˷�����" << iritz.com_of_mul << endl << endl;
						if (iritz.com_of_mul < best) {
							best = iritz.com_of_mul;
							best_r = r;
							best_step = cgstep;
						}
						output << nev << ", " << batch << ", " << r << ", " << cgstep << ", " << iritz.nIter << ", " << iritz.com_of_mul << endl;
					}
				}
				result << "����" << nev << "������������batchΪ" << batch << "��������Ҫ" << best
					<< "�γ˷����趨�������Ϊ" << best_r << "��cgstep�趨Ϊ" << best_step << endl << endl << endl;
			}
		}
		cout << "��" << matrixName << "ʹ�øĽ�Ritz��������" << endl;
		result.close();
		output.close();
#endif
		
		system("cls");
		++n_matrices;
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