#define EIGEN_USE_MKL_ALL
#include "mkl.h"

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
#include<mgmres.hpp>

using namespace std;
using namespace Eigen;

//������б�
//#define mLOBPCG_I
//#define mLOBPCG_II
//#define miRitz
//#define mBJD
#define mRitz

//�����׺
string suff = "";

time_t current;

//���־���
string matrices[1000] =
{	"bcsstk01",
	"bcsstk02",
	"bcsstk05",
	"bcsstk07",
	/*"bcsstk08",
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
	"sym-pos/Flan_1565"*/
};

#define min(x,y) (((x) < (y)) ? (x) : (y))

int main() {
	
	Eigen::initParallel();

	int n_matrices = 0;
	string method;
	
	//��Ҫ��ʱ�������
	//	srand((unsigned)time(NULL));
	
	struct param{
		int nev, batch, cg, res;
	};

	vector<param> LOBIparams = {
		{30, 10, 50, 0},
		{10, 10, 30, 0},
		{10, 10, 40, 0},
		{20, 10, 20, 0},
		{30, 10, 20, 0},
		{30, 20, 20, 0},
		{30, 30, 20, 0}
	};
	vector<param> LOBIIparams = {
		{30, 10, 60, 0},
		{10, 10, 30, 0},
		{10, 10, 40, 0},
		{20, 10, 20, 0},
		{30, 10, 20, 0},
		{30, 20, 20, 0},
		{30, 30, 20, 0}
	};
	vector<param> IRparams = {
		{10, 10, 20, 2},
		{10, 10, 20, 3},
		{10, 10, 20, 4},
		{10, 10, 30, 3},
		{10, 10, 40, 3},
		{20, 10, 20, 3},
		{30, 10, 20, 3},
		{30, 20, 20, 3},
		{30, 30, 20, 3}
	};
	vector<param> BJDparams = {
		{10, 10, 20, 10},
		{10, 10, 30, 10},
		{10, 10, 40, 10},
		{10, 10, 20, 20},
		{10, 10, 20, 30},
		{20, 10, 20, 10},
		{30, 10, 20, 10},
		{30, 20, 20, 10},
		{30, 30, 20, 10}
	};
	vector<param> Ritzparams = {
		/*{10, 10, 0, 3},
		{10, 10, 0, 4},
		{10, 10, 0, 5},
		{20, 10, 0, 3},
		{30, 10, 0, 3},*/
		{30, 20, 0, 3},
		{30, 30, 0, 3}
	};

	while (matrices[n_matrices].length() != 0) {
		
		//��A����
		string matrixName = matrices[n_matrices];
		SparseMatrix<double, RowMajor, __int64> A;
		if (matrixName.length() > 0)
			A = mtxio::getSparseMatrix("./matrix/" + matrixName + ".mtx");
		else
			A = mtxio::getSparseMatrix("./matrix/bcsstk01.mtx");

		//TODO B���õ�λ��
		SparseMatrix<double, RowMajor, __int64> B(A.rows(), A.cols());
		B.reserve(A.rows());
		for (int i = 0; i < A.rows(); ++i)
			B.insert(i, i) = 1;

		cout << "�������" << matrixName << "....................." << endl;

#ifdef mLOBPCG_I
		ofstream LP1output;
		ofstream LP1result;
		method = "LOBPCG_I";
		fstream_prepare(LP1result, LP1output, A, matrixName, method, suff);
		LP1output << "nev, batch, cgstep, iter, multi, time" << endl;
		for (int i = 0; i < LOBIparams.size(); ++i) {
			int nev = LOBIparams[i].nev;
			int batch = LOBIparams[i].batch;
			int cgstep = LOBIparams[i].cg;
			
			if (A.rows() / nev < 3)
				continue;
			if (batch > nev)
				continue;
			if (A.rows() / cgstep < 2)
				continue;
						
			//(SparseMatrix<double, RowMajor, __int64>& A, SparseMatrix<double, RowMajor, __int64>& B, int nev, int cgstep)
			LP1result << "LOBPCG_Iִ�в�����" << endl << "����ֵ��" << nev << "����batch��СΪ" << batch << "�����CG��������" << cgstep << "��" << endl;
			cout << "LOBPCG_Iִ�в�����" << endl << "����ֵ��" << nev << "����batch��СΪ" << batch << "�����CG��������" << cgstep << "��" << endl;
			LOBPCG_I_Batch LP1(A, B, nev, cgstep, batch);
			LP1.compute();

			for (int i = 0; i < LP1.eigenvalues.size(); ++i) {
				LP1result << "��" << i + 1 << "������ֵ��" << LP1.eigenvalues[i] << "��";
				//cout << "��" << i + 1 << "������������" << LP1.eigenvectors.col(i).transpose() << endl;
				LP1result << "�����" << (A * LP1.eigenvectors.col(i) - LP1.eigenvalues[i] * B * LP1.eigenvectors.col(i)).norm() / (A * LP1.eigenvectors.col(i)).norm() << endl;
			}
			LP1result << "LOBPCG_I����������" << LP1.nIter << endl;
			LP1result << "LOBPCG_I�˷�������" << LP1.com_of_mul << endl;
			LP1result << "LOBPCG_I����ʱ�䣺" << LP1.end_time - LP1.start_time << "��" << endl << endl;
						
			LP1output << nev << ", " << batch << ", " << cgstep << ", " << LP1.nIter << ", " << LP1.com_of_mul << ", " << LP1.end_time - LP1.start_time << endl;
			time_t now = time(&now);
			if (totalTimeCheck(current, now))
				break;
		}
		cout << "��" << matrixName << "ʹ��LOBPCG_I������" << endl;
		LP1result.close();
		LP1output.close();
#endif

#ifdef mLOBPCG_II
		ofstream LP2output;
		ofstream LP2result;
		method = "LOBPCG_II";
		fstream_prepare(LP2result, LP2output, A, matrixName, method, suff);
		LP2output << "nev, batch, cgstep, iter, multi, time" << endl;
		for (int i = 0; i < LOBIIparams.size(); ++i) {
			int nev = LOBIIparams[i].nev;
			int batch = LOBIIparams[i].batch;
			int cgstep = LOBIIparams[i].cg;

			if (A.rows() / nev < 3)
				continue;
			if (batch > nev)
				continue;
			if (A.rows() / cgstep < 2)
				continue;

			//(SparseMatrix<double, RowMajor, __int64>& A, SparseMatrix<double, RowMajor, __int64>& B, int nev, int cgstep)
			LP2result << "LOBPCG_IIִ�в�����" << endl << "����ֵ��" << nev << "����batch��СΪ" << batch << "�����CG��������" << cgstep << "��" << endl;
			cout << "LOBPCG_IIִ�в�����" << endl << "����ֵ��" << nev << "����batch��СΪ" << batch << "�����CG��������" << cgstep << "��" << endl;
			LOBPCG_II_Batch LP2(A, B, nev, cgstep, batch);
			LP2.compute();

			for (int i = 0; i < LP2.eigenvalues.size(); ++i) {
				LP2result << "��" << i + 1 << "������ֵ��" << LP2.eigenvalues[i] << "��";
				//cout << "��" << i + 1 << "������������" << LP1.eigenvectors.col(i).transpose() << endl;
				LP2result << "�����" << (A * LP2.eigenvectors.col(i) - LP2.eigenvalues[i] * B * LP2.eigenvectors.col(i)).norm() / (A * LP2.eigenvectors.col(i)).norm() << endl;
			}
			LP2result << "LOBPCG_II����������" << LP2.nIter << endl;
			LP2result << "LOBPCG_II�˷�������" << LP2.com_of_mul << endl;
			LP2result << "LOBPCG_II����ʱ�䣺" << LP2.end_time - LP2.start_time << "��" << endl << endl;
						
			LP2output << nev << ", " << batch << ", " << cgstep << ", " << LP2.nIter << ", " << LP2.com_of_mul << ", " << LP2.end_time - LP2.start_time << endl;
			time_t now = time(&now);
			if (totalTimeCheck(current, now))
				break;
		}
		cout << "��" << matrixName << "ʹ��LOBPCG_II������" << endl;
		LP2result.close();
		LP2output.close();
#endif

#ifdef miRitz
		ofstream IRoutput;
		ofstream IRresult;
		method = "IterRitz";
		fstream_prepare(IRresult, IRoutput, A, matrixName, method, suff);
		IRoutput << "nev, batch, r, cgstep, iter, multi, time" << endl;
		for (int i = 0; i < IRparams.size(); ++i) {
			int nev = IRparams[i].nev;
			int batch = IRparams[i].batch;
			int cgstep = IRparams[i].cg;
			int r = IRparams[i].res;

			if (nev < batch)
				continue;
			if (A.rows() / nev < 3)
				continue;
			if (A.rows() / (batch * r) < 2)
				continue;
			if (A.rows() / cgstep < 2)
				continue;
			
			//(SparseMatrix<double, RowMajor, __int64>& A, SparseMatrix<double, RowMajor, __int64>& B, int nev, int cgstep, int q, int r) 
			IRresult << "�Ľ�Ritz��ִ�в�����" << endl << "����ֵ��" << nev << "����batch��С��" << batch << "��Ritz������չ������" << r << ",���CG��������" << cgstep << endl;
			cout << "�Ľ�Ritz��ִ�в�����" << endl << "����ֵ��" << nev << "����batch��С��" << batch << "��Ritz������չ������" << r << ",���CG��������" << cgstep << endl;
			IterRitz iritz(A, B, nev, cgstep, batch, r);
			iritz.compute();

			for (int i = 0; i < iritz.eigenvalues.size(); ++i) {
				IRresult << "��" << i + 1 << "������ֵ��" << iritz.eigenvalues[i] << "��";
				//cout << "��" << i + 1 << "������������" << iritz.eigenvectors.col(i).transpose() << endl;
				IRresult << "�����" << (A * iritz.eigenvectors.col(i) - iritz.eigenvalues[i] * B * iritz.eigenvectors.col(i)).norm() / (A * iritz.eigenvectors.col(i)).norm() << endl;
			}
			IRresult << "�Ľ�Ritz����������" << iritz.nIter << endl;
			IRresult << "�Ľ�Ritz���˷�����" << iritz.com_of_mul << endl;
			IRresult << "�Ľ�Ritz������ʱ�䣺" << iritz.end_time - iritz.start_time << "��" << endl << endl;
							
			IRoutput << nev << ", " << batch << ", " << r << ", " << cgstep << ", " << iritz.nIter << ", " << iritz.com_of_mul << ", " << iritz.end_time - iritz.start_time << endl;
			time_t now = time(&now);
			if (totalTimeCheck(current, now))
				break;
		}
		cout << "��" << matrixName << "ʹ�øĽ�Ritz��������" << endl;
		IRresult.close();
		IRoutput.close();
#endif

#ifdef mBJD
		ofstream BJDoutput;
		ofstream BJDresult;
		method = "BJD";
		fstream_prepare(BJDresult, BJDoutput, A, matrixName, method, suff);
		BJDoutput << "nev, batch, restart, gmres_size, iter, nRestart, multi, time" << endl;
		for (int i = 0; i < BJDparams.size(); ++i) {
			int nev = BJDparams[i].nev;
			int batch = BJDparams[i].batch;
			int gmres_size = BJDparams[i].cg;
			int restart = BJDparams[i].res;
				
			if (A.rows() / nev < 3)
				continue;
			if (batch > nev)
				continue;
			if (A.rows() / (batch * restart) < 2)
				continue;
								
			//(SparseMatrix<double, RowMajor, __int64>& A, SparseMatrix<double, RowMajor, __int64>& B, int nev, int restart, int batch, int gmres_size, int gmres_restart)
			BJDresult << "��J-Dִ�в�����" << endl << "����ֵ��" << nev << "��������������" << restart << "��batch��С��" << batch << endl << "��GMRES��չ�ռ��С��" << gmres_size << endl;
			cout << "��J-Dִ�в�����" << endl << "����ֵ��" << nev << "��������������" << restart << "��batch��С��" << batch << endl << "��GMRES��չ�ռ��С��" << gmres_size << endl;
			BJD bjd(A, B, nev, restart, batch, gmres_size, 1);
			bjd.compute();

			for (int i = 0; i < bjd.eigenvalues.size(); ++i) {
				BJDresult << "��" << i + 1 << "������ֵ��" << bjd.eigenvalues[i] << "��";
				//cout << "��" << i + 1 << "������������" << LOBPCG.eigenvectors.col(i).transpose() << endl;
				BJDresult << "�����" << (A * bjd.eigenvectors.col(i) - bjd.eigenvalues[i] * B * bjd.eigenvectors.col(i)).norm() / (A * bjd.eigenvectors.col(i)).norm() << endl;
			}
			BJDresult << "��J-D��������" << bjd.nRestart << endl;
			BJDresult << "��J-D��������" << bjd.nIter << endl;
			BJDresult << "��J-D�˷�����" << bjd.com_of_mul << endl;
			BJDresult << "��J-D����ʱ�䣺" << bjd.end_time - bjd.start_time << "��" << endl << endl;

			BJDoutput << nev << ", " << batch << ", " << restart << ", " << gmres_size << ", " << bjd.nIter << ", " << bjd.nRestart << ", "
				<< bjd.com_of_mul << ", " << bjd.end_time - bjd.start_time << endl;
			time_t now = time(&now);
			if (totalTimeCheck(current, now))
				break;
		}
		cout << "��" << matrixName << "ʹ�ÿ�J-D������" << endl;
		BJDresult.close();
		BJDoutput.close();
#endif

#ifdef mRitz
		ofstream Ritzoutput;
		ofstream Ritzresult;
		method = "Ritz";
		fstream_prepare(Ritzresult, Ritzoutput, A, matrixName, method, suff);
		Ritzoutput << "nev, batch, r, iter, multi, time" << endl;
		for (int i = 0; i < Ritzparams.size(); ++i) {
			int nev = Ritzparams[i].nev;
			int batch = Ritzparams[i].batch;
			int r = Ritzparams[i].res;

			if (A.rows() / nev < 3)
				continue;
			if (batch > nev)
				continue;
			if (A.rows() / (batch * r) < 2)
				continue;
	
			//(SparseMatrix<double, RowMajor, __int64>& A, SparseMatrix<double, RowMajor, __int64>& B, int nev, int cgstep, int q, int r) 
			Ritzresult << "Ritz��ִ�в�����" << endl << "����ֵ��" << nev << "����batch��С��" << batch << "��Ritz������չ������" << r << endl;
			cout << "Ritz��ִ�в�����" << endl << "����ֵ��" << nev << "����batch��С��" << batch << "��Ritz������չ������" << r << endl;
			Ritz ritz(A, B, nev, 0, batch, r);
			ritz.compute();
			cout << ritz.com_of_mul << endl;
			system("pause");

			for (int i = 0; i < ritz.eigenvalues.size(); ++i) {
				Ritzresult << "��" << i + 1 << "������ֵ��" << ritz.eigenvalues[i] << "��";
				//cout << "��" << i + 1 << "������������" << ritz.eigenvectors.col(i).transpose() << endl;
				Ritzresult << "�����" << (A * ritz.eigenvectors.col(i) - ritz.eigenvalues[i] * B * ritz.eigenvectors.col(i)).norm() / (A * ritz.eigenvectors.col(i)).norm() << endl;
			}
			Ritzresult << "Ritz����������" << ritz.nIter << endl;
			Ritzresult << "Ritz���˷�����" << ritz.com_of_mul << endl;
			Ritzresult << "Ritz������ʱ�䣺" << ritz.end_time - ritz.start_time << "��" << endl << endl;
							
			Ritzoutput << nev << ", " << batch << ", " << r << ", " << ritz.nIter << ", " << ritz.com_of_mul << ", " << ritz.end_time - ritz.start_time << endl;
			time_t now = time(&now);
			if (totalTimeCheck(current, now))
				break;
		}
		cout << "��" << matrixName << "ʹ��Ritz��������" << endl;
		Ritzresult.close();
		Ritzoutput.close();
#endif
		
		system("cls");
		++n_matrices;
	}
}