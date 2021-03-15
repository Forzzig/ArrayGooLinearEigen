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

	//各种矩阵
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

		//需要的时候开随机化
		//	srand((unsigned)time(NULL));

		//给人看的就不用科学计数法了
		//cout << scientific << setprecision(16);

		//读A矩阵
		string matrixName = matrices[n_matrices];
		SparseMatrix<double> A;
		if (matrixName.length() > 0)
			A = mtxio::getSparseMatrix("./matrix/" + matrixName + ".mtx");
		else
			A = mtxio::getSparseMatrix("./matrix/bcsstk01.mtx");

		//TODO B先用单位阵
		SparseMatrix<double> B(A.rows(), A.cols());
		B.reserve(A.rows());
		for (int i = 0; i < A.rows(); ++i)
			B.insert(i, i) = 1;
		

		/*cout << "A-------------------------" << endl << A << endl;
		cout << "B-------------------------" << endl << B << endl;*/
		//system("pause");

		cout << "正在求解" << matrixName << "....................." << endl;

		cout << "对" << matrixName << "使用LOBPCG_I....................." << endl;
		
		result.open(matrixName + "-LOBPCG_I.txt");
		result << "矩阵阶数：" << A.rows() << endl;
		result << "非零元数：" << A.nonZeros() << endl << endl;
		result << matrixName + "，LOBPCG_I开始求解........................................." << endl;
		for (int nev = 10; nev <= 50; nev += 10) {
			if (A.rows() / nev < 3)
				break;
			for (int cgstep = 10; cgstep <= 50; cgstep += 10) {
				if (A.rows() / cgstep < 2)
					break;
				//(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep)
				result << "LOBPCG_I执行参数：" << endl << "特征值：" << nev << "个，最大CG迭代步：" << cgstep << "次" << endl;
				cout << "LOBPCG_I执行参数：" << endl << "特征值：" << nev << "个，最大CG迭代步：" << cgstep << "次" << endl;
				LOBPCG_I LP1(A, B, nev, cgstep);
				LP1.compute();

				for (int i = 0; i < LP1.eigenvalues.size(); ++i) {
					result << "第" << i + 1 << "个特征值：" << LP1.eigenvalues[i] << "，";
					//cout << "第" << i + 1 << "个特征向量：" << LP1.eigenvectors.col(i).transpose() << endl;
					result << "相对误差：" << (A * LP1.eigenvectors.col(i) - LP1.eigenvalues[i] * B * LP1.eigenvectors.col(i)).norm() / (A * LP1.eigenvectors.col(i)).norm() << endl;
				}
				result << "LOBPCG_I迭代次数：" << LP1.nIter << endl;
				result << "LOBPCG_I乘法次数：" << LP1.com_of_mul << endl << endl;
			}
		}
		cout << "对" << matrixName << "使用LOBPCG_I结束。" << endl;
		result.close();

		result.open(matrixName + "-LOBPCG_II.txt");
		result << "矩阵阶数：" << A.rows() << endl;
		result << "非零元数：" << A.nonZeros() << endl << endl;
		cout << "对" << matrixName << "使用LOBPCG_II....................." << endl;
		result << matrixName + "，LOBPCG_II开始求解........................................." << endl;
		for (int nev = 10; nev <= 50; nev += 10) {
			if (A.rows() / nev < 3)
				break;
			for (int cgstep = 10; cgstep <= 50; cgstep += 10) {
				if (A.rows() / cgstep < 2)
					break;
				//(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep)
				result << "LOBPCG_II执行参数：" << endl << "特征值：" << nev << "个，最大CG迭代步：" << cgstep << "次" << endl;
				cout << "LOBPCG_II执行参数：" << endl << "特征值：" << nev << "个，最大CG迭代步：" << cgstep << "次" << endl;
				LOBPCG_II LP2(A, B, nev, cgstep);
				LP2.compute();

				for (int i = 0; i < LP2.eigenvalues.size(); ++i) {
					result << "第" << i + 1 << "个特征值：" << LP2.eigenvalues[i] << "，";
					//cout << "第" << i + 1 << "个特征向量：" << LP1.eigenvectors.col(i).transpose() << endl;
					result << "相对误差：" << (A * LP2.eigenvectors.col(i) - LP2.eigenvalues[i] * B * LP2.eigenvectors.col(i)).norm() / (A * LP2.eigenvectors.col(i)).norm() << endl;
				}
				result << "LOBPCG_II迭代次数：" << LP2.nIter << endl;
				result << "LOBPCG_II乘法次数：" << LP2.com_of_mul << endl << endl;
			}
		}
		cout << "对" << matrixName << "使用LOBPCG_II结束。" << endl;
		result.close();

		result.open(matrixName + "-BJD.txt");
		result << "矩阵阶数：" << A.rows() << endl;
		result << "非零元数：" << A.nonZeros() << endl << endl;
		cout << "对" << matrixName << "使用块J-D....................." << endl;
		result << matrixName + "，块J-D开始求解........................................." << endl;
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
							result << "块J-D执行参数：" << endl << "特征值：" << nev << "个，重启步数：" << restart << "，batch大小：" << batch << endl << 
								"    GMRES总迭代步数（扩展空间乘重启次数）：" << gmres_size * gmres_restart << "，GMRES扩展空间大小：" << gmres_size << endl;
							cout << "块J-D执行参数：" << endl << "特征值：" << nev << "个，重启步数：" << restart << "，batch大小：" << batch << endl <<
								"    GMRES总迭代步数（扩展空间乘重启次数）：" << gmres_size * gmres_restart << "，GMRES扩展空间大小：" << gmres_size << endl;
							BJD bjd(A, B, nev, gmres_size * gmres_restart, restart, batch, gmres_size);
							//L_GMRES(SparseMatrix<double>& A, SparseMatrix<double>& B, Derived_rhs& b, Derived_ss& U, Derived_sol& X, Derived_eval& lam, int m)
							bjd.compute();

							for (int i = 0; i < bjd.eigenvalues.size(); ++i) {
								result << "第" << i + 1 << "个特征值：" << bjd.eigenvalues[i] << "，";
								//cout << "第" << i + 1 << "个特征向量：" << LOBPCG.eigenvectors.col(i).transpose() << endl;
								result << "相对误差：" << (A * bjd.eigenvectors.col(i) - bjd.eigenvalues[i] * B * bjd.eigenvectors.col(i)).norm() / (A * bjd.eigenvectors.col(i)).norm() << endl;
							}
							result << "块J-D迭代次数" << bjd.nIter << endl;
							result << "块J-D乘法次数" << bjd.com_of_mul << endl << endl;
						}
					}
				}
			}
		}
		cout << "对" << matrixName << "使用块J-D结束。" << endl;
		result.close();

		result.open(matrixName + "-Ritz.txt");
		result << "矩阵阶数：" << A.rows() << endl;
		result << "非零元数：" << A.nonZeros() << endl << endl;
		cout << "对" << matrixName << "使用迭代Ritz法....................." << endl;
		result << matrixName + "，迭代Ritz法开始求解........................................." << endl;
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
						result << "迭代Ritz法执行参数：" << endl << "特征值：" << nev << "个，batch大小：" << batch << "，Ritz向量扩展个数：" << r << ",最大CG迭代步：" << cgstep << endl;
						cout << "迭代Ritz法执行参数：" << endl << "特征值：" << nev << "个，batch大小：" << batch << "，Ritz向量扩展个数：" << r << ",最大CG迭代步：" << cgstep << endl;
						Ritz ritz(A, B, nev, cgstep, batch, r);
						ritz.compute();

						for (int i = 0; i < ritz.eigenvalues.size(); ++i) {
							result << "第" << i + 1 << "个特征值：" << ritz.eigenvalues[i] << "，";
							//cout << "第" << i + 1 << "个特征向量：" << ritz.eigenvectors.col(i).transpose() << endl;
							result << "相对误差：" << (A * ritz.eigenvectors.col(i) - ritz.eigenvalues[i] * B * ritz.eigenvectors.col(i)).norm() / (A * ritz.eigenvectors.col(i)).norm() << endl;
						}
						result << "迭代Ritz法迭代次数" << ritz.nIter << endl;
						result << "迭代Ritz法乘法次数" << ritz.com_of_mul << endl;
					}
				}
			}
		}
		cout << "对" << matrixName << "使用迭代Ritz法结束。" << endl;
		result.close();

		result.open(matrixName + "-iRitz.txt");
		result << "矩阵阶数：" << A.rows() << endl;
		result << "非零元数：" << A.nonZeros() << endl << endl;
		cout << "对" << matrixName << "使用改进Ritz法....................." << endl;
		result << matrixName + "，改进Ritz法开始求解........................................." << endl;
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
						result << "改进Ritz法执行参数：" << endl << "特征值：" << nev << "个，batch大小：" << batch << "，Ritz向量扩展个数：" << r << ",最大CG迭代步：" << cgstep << endl;
						cout << "改进Ritz法执行参数：" << endl << "特征值：" << nev << "个，batch大小：" << batch << "，Ritz向量扩展个数：" << r << ",最大CG迭代步：" << cgstep << endl;
						IterRitz iritz(A, B, nev, cgstep, batch, r);
						iritz.compute();

						for (int i = 0; i < iritz.eigenvalues.size(); ++i) {
							result << "第" << i + 1 << "个特征值：" << iritz.eigenvalues[i] << "，";
							//cout << "第" << i + 1 << "个特征向量：" << iritz.eigenvectors.col(i).transpose() << endl;
							result << "相对误差：" << (A * iritz.eigenvectors.col(i) - iritz.eigenvalues[i] * B * iritz.eigenvectors.col(i)).norm() / (A * iritz.eigenvectors.col(i)).norm() << endl;
						}
						result << "改进Ritz法迭代次数" << iritz.nIter << endl;
						result << "改进Ritz法乘法次数" << iritz.com_of_mul << endl;
					}
				}
			}
		}
		cout << "对" << matrixName << "使用改进Ritz法结束。" << endl;
		result.close();
		system("cls");
		//system("pause");

		///*cout << "原始GCG开始求解..." << endl;
		//GCG_sv gsv(A, B, 10, 40, 10);
		//gsv.compute();*/

		//cout << "LOBPCG开始求解..." << endl;
		//LOBPCG_solver LOBPCG(A, B, 20, 40);
		//LOBPCG.compute();

		//cout << "JD开始求解..." << endl;
		////(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int restart, int batch, int gmres_size)
		//JD jd(A, B, 20, 100, 10, 5, 10);
		///*jd.compute();*/


		
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



		//for (int i = 0; i < jd.eigenvalues.size(); ++i) {
		//	cout << "第" << i + 1 << "个特征值：" << jd.eigenvalues[i] << endl;
		//	//cout << "第" << i + 1 << "个特征向量：" << LOBPCG.eigenvectors.col(i).transpose() << endl;
		//	cout << (A * jd.eigenvectors.col(i) - jd.eigenvalues[i] * B * jd.eigenvectors.col(i)).norm() / (A * jd.eigenvectors.col(i)).norm() << endl;
		//}
		//cout << "JD迭代次数" << jd.nIter << endl;

	}
	system("pause");
}