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

//各种矩阵
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

#ifdef mLOBPCG_I
		cout << "对" << matrixName << "使用LOBPCG_I....................." << endl;
		
		result.open(matrixName + "-LOBPCG_I.txt");
		result << "矩阵阶数：" << A.rows() << endl;
		result << "非零元数：" << A.nonZeros() << endl << endl;
		result << matrixName + "，LOBPCG_I开始求解........................................." << endl;
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
					result << "LOBPCG_I执行参数：" << endl << "特征值：" << nev << "个，batch大小为" << batch << "，最大CG迭代步：" << cgstep << "次" << endl;
					cout << "LOBPCG_I执行参数：" << endl << "特征值：" << nev << "个，batch大小为" << batch << "，最大CG迭代步：" << cgstep << "次" << endl;
					LOBPCG_I_Batch LP1(A, B, nev, cgstep, batch);
					LP1.compute();

					for (int i = 0; i < LP1.eigenvalues.size(); ++i) {
						result << "第" << i + 1 << "个特征值：" << LP1.eigenvalues[i] << "，";
						//cout << "第" << i + 1 << "个特征向量：" << LP1.eigenvectors.col(i).transpose() << endl;
						result << "相对误差：" << (A * LP1.eigenvectors.col(i) - LP1.eigenvalues[i] * B * LP1.eigenvectors.col(i)).norm() / (A * LP1.eigenvectors.col(i)).norm() << endl;
					}
					result << "LOBPCG_I迭代次数：" << LP1.nIter << endl;
					result << "LOBPCG_I乘法次数：" << LP1.com_of_mul << endl << endl;
					if (LP1.com_of_mul < best) {
						best = LP1.com_of_mul;
						best_step = cgstep;
					}
					output << nev << ", " << batch << ", " << cgstep << ", " << LP1.nIter << ", " << LP1.com_of_mul << endl;
				}
				result << "计算" << nev << "个特征向量，batch大小为" << batch << "，最少需要" << best << "次乘法，cgstep设定为" << best_step << endl << endl << endl;
			}
		}
		cout << "对" << matrixName << "使用LOBPCG_I结束。" << endl;
		result.close();
		output.close();
#endif

#ifdef mLOBPCG_II
		result.open(matrixName + "-LOBPCG_II.txt");
		result << "矩阵阶数：" << A.rows() << endl;
		result << "非零元数：" << A.nonZeros() << endl << endl;
		cout << "对" << matrixName << "使用LOBPCG_II....................." << endl;
		result << matrixName + "，LOBPCG_II开始求解........................................." << endl;
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
					result << "LOBPCG_II执行参数：" << endl << "特征值：" << nev << "个，batch大小为" << batch << "，最大CG迭代步：" << cgstep << "次" << endl;
					cout << "LOBPCG_II执行参数：" << endl << "特征值：" << nev << "个，batch大小为" << batch << "，最大CG迭代步：" << cgstep << "次" << endl;
					LOBPCG_II_Batch LP2(A, B, nev, cgstep, batch);
					LP2.compute();

					for (int i = 0; i < LP2.eigenvalues.size(); ++i) {
						result << "第" << i + 1 << "个特征值：" << LP2.eigenvalues[i] << "，";
						//cout << "第" << i + 1 << "个特征向量：" << LP1.eigenvectors.col(i).transpose() << endl;
						result << "相对误差：" << (A * LP2.eigenvectors.col(i) - LP2.eigenvalues[i] * B * LP2.eigenvectors.col(i)).norm() / (A * LP2.eigenvectors.col(i)).norm() << endl;
					}
					result << "LOBPCG_II迭代次数：" << LP2.nIter << endl;
					result << "LOBPCG_II乘法次数：" << LP2.com_of_mul << endl << endl;
					if (LP2.com_of_mul < best) {
						best = LP2.com_of_mul;
						best_step = cgstep;
					}
					output << nev << ", " << batch << ", " << cgstep << ", " << LP2.nIter << ", " << LP2.com_of_mul << endl;
				}
				result << "计算" << nev << "个特征向量，batch大小为" << batch << "，最少需要" << best << "次乘法，cgstep设定为" << best_step << endl << endl << endl;
			}
		}
		cout << "对" << matrixName << "使用LOBPCG_II结束。" << endl;
		result.close();
		output.close();
#endif
#ifdef mBJD
		result.open(matrixName + "-BJD.txt");
		result << "矩阵阶数：" << A.rows() << endl;
		result << "非零元数：" << A.nonZeros() << endl << endl;
		cout << "对" << matrixName << "使用块J-D....................." << endl;
		result << matrixName + "，块J-D开始求解........................................." << endl;
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
				result << "计算" << nev << "个特征向量，batch为" << batch << "，最少需要" << best
					<< "次乘法，设定为迭代" << best_restart << "次重启，gmres设定扩展空间大小" << best_gmres_size << "，总迭代步数" << best_gmres_size * best_gmres_restart << endl << endl << endl;
			}
		}
		cout << "对" << matrixName << "使用块J-D结束。" << endl;
		result.close();
		output.close();
#endif
#ifdef mRitz
		result.open(matrixName + "-Ritz.txt");
		result << "矩阵阶数：" << A.rows() << endl;
		result << "非零元数：" << A.nonZeros() << endl << endl;
		cout << "对" << matrixName << "使用迭代Ritz法....................." << endl;
		result << matrixName + "，迭代Ritz法开始求解........................................." << endl;
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
						result << "迭代Ritz法乘法次数" << ritz.com_of_mul << endl << endl;
						if (ritz.com_of_mul < best) {
							best = ritz.com_of_mul;
							best_r = r;
							best_step = cgstep;
						}
						output << nev << ", " << batch << ", " << r << ", " << cgstep << ", " << ritz.nIter << ", " << ritz.com_of_mul << endl;
					}
				}
				result << "计算" << nev << "个特征向量，batch为" << batch << "，最少需要" << best
					<< "次乘法，设定迭代深度为" << best_r << "，cgstep设定为" << best_step << endl << endl << endl;
			}
		}
		cout << "对" << matrixName << "使用迭代Ritz法结束。" << endl;
		result.close();
		output.close();
#endif
#ifdef miRitz
		result.open(matrixName + "-iRitz.txt");
		result << "矩阵阶数：" << A.rows() << endl;
		result << "非零元数：" << A.nonZeros() << endl << endl;
		cout << "对" << matrixName << "使用改进Ritz法....................." << endl;
		result << matrixName + "，改进Ritz法开始求解........................................." << endl;
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
						result << "改进Ritz法乘法次数" << iritz.com_of_mul << endl << endl;
						if (iritz.com_of_mul < best) {
							best = iritz.com_of_mul;
							best_r = r;
							best_step = cgstep;
						}
						output << nev << ", " << batch << ", " << r << ", " << cgstep << ", " << iritz.nIter << ", " << iritz.com_of_mul << endl;
					}
				}
				result << "计算" << nev << "个特征向量，batch为" << batch << "，最少需要" << best
					<< "次乘法，设定迭代深度为" << best_r << "，cgstep设定为" << best_step << endl << endl << endl;
			}
		}
		cout << "对" << matrixName << "使用改进Ritz法结束。" << endl;
		result.close();
		output.close();
#endif
		
		system("cls");
		++n_matrices;
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