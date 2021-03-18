#include<iostream>
#include<Eigen\Sparse>
#include<mtxio.h>
#include<LOBPCG_I_Batch.h>
#include<LOBPCG_II_Batch.h>
#include<IterRitz.h>
#include<Ritz.h>
#include<BJD.h>
#include<ctime>
#include<chrono>
#include<cstdlib>
#include<fstream>
#include<string>
#include<TimeControl.h>
#include<output_helper.h>

using namespace std;
using namespace Eigen;

fstream output;
fstream result;

//������б�
#define mLOBPCG_I
#define mLOBPCG_II
#define miRitz
#define mBJD
#define mRitz

//�����׺
string suff = "";

time_t current;

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
	"1138_bus",

	"sym-pos/apache1",
	"sym-pos/ct20stif",
	"sym-pos/oilpan",
	"sym-pos/apache2",
	"sym-pos/shipsec8",
	"sym-pos/ship_003",
	"sym-pos/shipsec5",
	"sym-pos/crankseg_1",
	"sym-pos/bmw7st_1",
	"sym-pos/m_t1",
	"sym-pos/x104",
	"sym-pos/hood",
	"sym-pos/crankseg_2",
	"sym-pos/pwtk",
	"sym-pos/bmwcra_1",
	"sym-pos/msdoor",
	"sym-pos/StocF-1465",
	"sym-pos/Fault_639",
	"sym-pos/Emilia_923",
	"sym-pos/inline_1",
	"sym-pos/ldoor",
	"sym-pos/Hook_1498",
	"sym-pos/Geo_1438",
	"sym-pos/Serena",
	"sym-pos/audikw_1",
	"sym-pos/Flan_1565"
};

int main() {

	int n_matrices = 0;
	string method;
	
	//��Ҫ��ʱ�������
	//	srand((unsigned)time(NULL));
	
	while (matrices[n_matrices].length() != 0) {
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
		
		int cgrange = floor(log(A.rows() / 1000 + 1) / log(10)) * 6 + 10;
		int batchrange = 5;
		int nevrange = 20;
		int restartrange = floor(log(A.rows() / 1000 + 1) / log(10)) * 5 + 10;

		cout << "�������" << matrixName << "....................." << endl;

#ifdef mLOBPCG_I
		method = "LOBPCG_I";
		fstream_prepare(result, output, A, matrixName, method, suff);
		output << "nev, batch, cgstep, iter, multi" << endl;
		for (int nev = nevrange; nev <= nevrange; nev += 5) {
			if (A.rows() / nev < 3)
				break;
			for (int batch = batchrange; batch <= batchrange + 5; batch += 5) {
				if (batch > nev)
					break;
				long long best = LLONG_MAX;
				int best_step;
				//TODO �ʵ���һЩ
				for (int cgstep = cgrange + 10; cgstep <= cgrange + 30; cgstep += 10) {
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
					time_t now = time(&now);
					if (totalTimeCheck(current, now))
						break;
				}
				result << "����" << nev << "������������batch��СΪ" << batch << "��������Ҫ" << best << "�γ˷���cgstep�趨Ϊ" << best_step << endl << endl << endl;
				time_t now = time(&now);
				if (totalTimeCheck(current, now))
					break;
			}
			time_t now = time(&now);
			if (totalTimeCheck(current, now))
				break;
		}
		cout << "��" << matrixName << "ʹ��LOBPCG_I������" << endl;
		result.close();
		output.close();
#endif

#ifdef mLOBPCG_II
		method = "LOBPCG_II";
		fstream_prepare(result, output, A, matrixName, method, suff);
		output << "nev, batch, cgstep, iter, multi" << endl;
		for (int nev = nevrange; nev <= nevrange; nev += 5) {
			if (A.rows() / nev < 3)
				break;
			for (int batch = batchrange; batch <= batchrange + 5; batch += 5) {
				if (batch > nev)
					break;
				long long best = LLONG_MAX;
				int best_step;

				//TODO �ʵ���һЩ
				for (int cgstep = cgrange - 5; cgstep <= cgrange + 15; cgstep += 10) {
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
					time_t now = time(&now);
					if (totalTimeCheck(current, now))
						break;
				}
				result << "����" << nev << "������������batch��СΪ" << batch << "��������Ҫ" << best << "�γ˷���cgstep�趨Ϊ" << best_step << endl << endl << endl;
				time_t now = time(&now);
				if (totalTimeCheck(current, now))
					break;
			}
			time_t now = time(&now);
			if (totalTimeCheck(current, now))
				break;
		}
		cout << "��" << matrixName << "ʹ��LOBPCG_II������" << endl;
		result.close();
		output.close();
#endif

#ifdef miRitz
		method = "IterRitz";
		fstream_prepare(result, output, A, matrixName, method, suff);
		output << "nev, batch, r, cgstep, iter, multi" << endl;
		for (int nev = nevrange; nev <= nevrange; nev += 5) {
			for (int batch = batchrange; batch <= batchrange + 5; batch += 5) {
				if (nev < batch)
					break;
				if (A.rows() / nev < 3)
					break;

				long long best = LLONG_MAX;
				int best_r, best_step;
				for (int r = 2; r <= 4; ++r) {
					if (A.rows() / (batch * r) < 2)
						break;
					//TODO �Զ�һ���
					for (int cgstep = cgrange + 5; cgstep <= cgrange + 25; cgstep += 10) {
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
						time_t now = time(&now);
						if (totalTimeCheck(current, now))
							break;
					}
					time_t now = time(&now);
					if (totalTimeCheck(current, now))
						break;
				}
				result << "����" << nev << "������������batchΪ" << batch << "��������Ҫ" << best
					<< "�γ˷����趨�������Ϊ" << best_r << "��cgstep�趨Ϊ" << best_step << endl << endl << endl;
				time_t now = time(&now);
				if (totalTimeCheck(current, now))
					break;
			}
			time_t now = time(&now);
			if (totalTimeCheck(current, now))
				break;
		}
		cout << "��" << matrixName << "ʹ�øĽ�Ritz��������" << endl;
		result.close();
		output.close();
#endif

#ifdef mBJD
		method = "BJD";
		fstream_prepare(result, output, A, matrixName, method, suff);
		output << "nev, batch, restart, gmres_size, gmres_restart, gmres_step, iter, nRestart, multi" << endl;
		for (int nev = nevrange; nev <= nevrange; nev += 5) {
			if (A.rows() / nev < 3)
				break;
			for (int batch = batchrange; batch <= batchrange + 5; batch += 5) {
				if (batch > nev)
					break;
				long long best = LLONG_MAX;
				int best_gmres_size, best_gmres_restart, best_restart;

				for (int restart = restartrange; restart <= restartrange + 10; restart += 5) {
					if (A.rows() / (batch * restart) < 2)
						break;

					//TODO ������
					for (int gmres_size = cgrange / 2; gmres_size <= cgrange / 2 + 20; gmres_size += 10) {
						if (A.rows() / gmres_size < 2)
							break;
						for (int gmres_restart = 1; gmres_restart <= 1; ++gmres_restart) {
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
							result << "��J-D��������" << bjd.nRestart << endl;
							result << "��J-D��������" << bjd.nIter << endl;
							result << "��J-D�˷�����" << bjd.com_of_mul << endl << endl;
							if (bjd.com_of_mul < best) {
								best = bjd.com_of_mul;
								best_gmres_size = gmres_size;
								best_gmres_restart = gmres_restart;
								best_restart = restart;
							}
							output << nev << ", " << batch << ", " << restart << ", " << gmres_size << ", " << gmres_restart << ", " 
								<< gmres_size * gmres_restart << ", " << bjd.nIter << ", " << bjd.nRestart << ", " << bjd.com_of_mul << endl;
							time_t now = time(&now);
							if (totalTimeCheck(current, now))
								break;
						}
						time_t now = time(&now);
						if (totalTimeCheck(current, now))
							break;
					}
					time_t now = time(&now);
					if (totalTimeCheck(current, now))
						break;
				}
				result << "����" << nev << "������������batchΪ" << batch << "��������Ҫ" << best << "�γ˷����趨Ϊ����" << best_restart << 
					"��������gmres�趨��չ�ռ��С" << best_gmres_size << "���ܵ�������" << best_gmres_size * best_gmres_restart << endl << endl << endl;
				time_t now = time(&now);
				if (totalTimeCheck(current, now))
					break;
			}
			time_t now = time(&now);
			if (totalTimeCheck(current, now))
				break;
		}
		cout << "��" << matrixName << "ʹ�ÿ�J-D������" << endl;
		result.close();
		output.close();
#endif

#ifdef mRitz
		method = "Ritz";
		fstream_prepare(result, output, A, matrixName, method, suff);
		output << "nev, batch, r, cgstep, iter, multi" << endl;
		for (int nev = nevrange; nev <= nevrange; nev += 5) {
			if (A.rows() / nev < 3)
				break;
			for (int batch = batchrange; batch <= batchrange + 5; batch += 5) {
				if (batch > nev)
					break;

				long long best = LLONG_MAX;
				int best_r, best_step;
				for (int r = 2; r <= 4; ++r) {
					if (A.rows() / (batch * r) < 2)
						break;

					//TODO �Զ�һ���
					for (int cgstep = cgrange + 5; cgstep <= cgrange + 25; cgstep += 10) {
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
						time_t now = time(&now);
						if (totalTimeCheck(current, now))
							break;
					}
					time_t now = time(&now);
					if (totalTimeCheck(current, now))
						break;
				}
				result << "����" << nev << "������������batchΪ" << batch << "��������Ҫ" << best
					<< "�γ˷����趨�������Ϊ" << best_r << "��cgstep�趨Ϊ" << best_step << endl << endl << endl;
				time_t now = time(&now);
				if (totalTimeCheck(current, now))
					break;
			}
			time_t now = time(&now);
			if (totalTimeCheck(current, now))
				break;
		}
		cout << "��" << matrixName << "ʹ�õ���Ritz��������" << endl;
		result.close();
		output.close();
#endif
		
		system("cls");
		++n_matrices;
	}
}