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

//求解器列表
#define mLOBPCG_I
#define mLOBPCG_II
#define miRitz
#define mBJD
#define mRitz

//特殊后缀
string suff = "";

time_t current;

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
	
	//需要的时候开随机化
	//	srand((unsigned)time(NULL));
	
	while (matrices[n_matrices].length() != 0) {
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
		
		int cgrange = floor(log(A.rows() / 1000 + 1) / log(10)) * 6 + 10;
		int batchrange = 5;
		int nevrange = 20;
		int restartrange = floor(log(A.rows() / 1000 + 1) / log(10)) * 5 + 10;

		cout << "正在求解" << matrixName << "....................." << endl;

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
				//TODO 适当多一些
				for (int cgstep = cgrange + 10; cgstep <= cgrange + 30; cgstep += 10) {
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
					time_t now = time(&now);
					if (totalTimeCheck(current, now))
						break;
				}
				result << "计算" << nev << "个特征向量，batch大小为" << batch << "，最少需要" << best << "次乘法，cgstep设定为" << best_step << endl << endl << endl;
				time_t now = time(&now);
				if (totalTimeCheck(current, now))
					break;
			}
			time_t now = time(&now);
			if (totalTimeCheck(current, now))
				break;
		}
		cout << "对" << matrixName << "使用LOBPCG_I结束。" << endl;
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

				//TODO 适当少一些
				for (int cgstep = cgrange - 5; cgstep <= cgrange + 15; cgstep += 10) {
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
					time_t now = time(&now);
					if (totalTimeCheck(current, now))
						break;
				}
				result << "计算" << nev << "个特征向量，batch大小为" << batch << "，最少需要" << best << "次乘法，cgstep设定为" << best_step << endl << endl << endl;
				time_t now = time(&now);
				if (totalTimeCheck(current, now))
					break;
			}
			time_t now = time(&now);
			if (totalTimeCheck(current, now))
				break;
		}
		cout << "对" << matrixName << "使用LOBPCG_II结束。" << endl;
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
					//TODO 稍多一点点
					for (int cgstep = cgrange + 5; cgstep <= cgrange + 25; cgstep += 10) {
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
						time_t now = time(&now);
						if (totalTimeCheck(current, now))
							break;
					}
					time_t now = time(&now);
					if (totalTimeCheck(current, now))
						break;
				}
				result << "计算" << nev << "个特征向量，batch为" << batch << "，最少需要" << best
					<< "次乘法，设定迭代深度为" << best_r << "，cgstep设定为" << best_step << endl << endl << endl;
				time_t now = time(&now);
				if (totalTimeCheck(current, now))
					break;
			}
			time_t now = time(&now);
			if (totalTimeCheck(current, now))
				break;
		}
		cout << "对" << matrixName << "使用改进Ritz法结束。" << endl;
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

					//TODO 必须少
					for (int gmres_size = cgrange / 2; gmres_size <= cgrange / 2 + 20; gmres_size += 10) {
						if (A.rows() / gmres_size < 2)
							break;
						for (int gmres_restart = 1; gmres_restart <= 1; ++gmres_restart) {
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
							result << "块J-D重启轮数" << bjd.nRestart << endl;
							result << "块J-D迭代次数" << bjd.nIter << endl;
							result << "块J-D乘法次数" << bjd.com_of_mul << endl << endl;
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
				result << "计算" << nev << "个特征向量，batch为" << batch << "，最少需要" << best << "次乘法，设定为迭代" << best_restart << 
					"次重启，gmres设定扩展空间大小" << best_gmres_size << "，总迭代步数" << best_gmres_size * best_gmres_restart << endl << endl << endl;
				time_t now = time(&now);
				if (totalTimeCheck(current, now))
					break;
			}
			time_t now = time(&now);
			if (totalTimeCheck(current, now))
				break;
		}
		cout << "对" << matrixName << "使用块J-D结束。" << endl;
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

					//TODO 稍多一点点
					for (int cgstep = cgrange + 5; cgstep <= cgrange + 25; cgstep += 10) {
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
						time_t now = time(&now);
						if (totalTimeCheck(current, now))
							break;
					}
					time_t now = time(&now);
					if (totalTimeCheck(current, now))
						break;
				}
				result << "计算" << nev << "个特征向量，batch为" << batch << "，最少需要" << best
					<< "次乘法，设定迭代深度为" << best_r << "，cgstep设定为" << best_step << endl << endl << endl;
				time_t now = time(&now);
				if (totalTimeCheck(current, now))
					break;
			}
			time_t now = time(&now);
			if (totalTimeCheck(current, now))
				break;
		}
		cout << "对" << matrixName << "使用迭代Ritz法结束。" << endl;
		result.close();
		output.close();
#endif
		
		system("cls");
		++n_matrices;
	}
}