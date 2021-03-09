#include<LOBPCG_I.h>
#include<iostream>

using namespace std;
LOBPCG_I::LOBPCG_I(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep)
	: LinearEigenSolver(A, B, nev),
		storage(new double[A.rows() * nev * 3]),
		X(storage + A.rows() * nev, A.rows(), nev),
		P(storage + A.rows() * nev * 2, A.rows(), 0),
		W(storage, A.rows(), nev) {
	X = MatrixXd::Random(A.rows(), nev);
	orthogonalization(X, B);
	cout << "X初始化" << endl;

	//只有要求解的矩阵不变时才能在这初始化
	linearsolver.compute(A);
	linearsolver.setMaxIterations(cgstep);
	cout << "CG求解器准备完成..." << endl;
	cout << "初始化完成" << endl;
}


void LOBPCG_I::compute() {
	vector<int> cnv;
	MatrixXd eval, evec, tmp, tmpA, mu;

	//所有已计算出来的特征向量和待计算的都依次存在storage里，避免内存拷贝
	Map<MatrixXd> V(storage, A.rows(), X.cols() + W.cols()), vl(storage, 0, 0);
	while (true) {
		++nIter;
		cout << "迭代步：" << nIter << endl;
		tmpA = A * X;
		tmp = B * X;
		for (int i = 0; i < nev; ++i) {
			mu = X.col(i).transpose() * A * X.col(i);
			tmp.col(i) *= mu(0, 0);
		}

		//求解 A*W = A*X - mu*B*X， X为上一步的近似特征向量
		W = linearsolver.solve(tmpA - tmp);
		//预存上一步的近似特征向量
		tmp = X;
		/*cout << W.cols() << " " << X.cols() << " " << P.cols()<< endl;*/

		//因为不取出已收敛的向量，所以V从storage头部开始
		new (&V) Map<MatrixXd>(storage, A.rows(), W.cols() + X.cols() + P.cols());
		
		//全体做正交化，如果分块容易出问题
		int dep = orthogonalization(V, B);
		//正交化后会有线性相关项，剔除
		new (&V) Map<MatrixXd>(storage, A.rows(), W.cols() + X.cols() + P.cols() - dep);
		cout << V.cols() << endl;
		
		projection_RR(V, A, eval, evec);

		system("cls");
		cout << V.rows() << " " << V.cols() << " " << evec.rows() << " " << evec.cols() << endl;
		//子空间V投影下的新的近似特征向量
		X = V * evec.block(0, 0, V.cols(), nev);
		//P是上一步的近似特征向量
		new (&P) Map<MatrixXd>(storage + A.rows() * (W.cols() + X.cols()), A.rows(), nev);
		P = tmp;

		//设置v1是为了方便控制检查收敛的个数
		new (&vl) Map<MatrixXd>(&(eval(0, 0)), nev, 1);
		
		cnv = conv_check(vl, X, 0);
		cout << "已收敛特征向量个数：" << cnv.size() << endl;

		if (cnv.size() >= nev) {
			eigenvectors = X;
			for (int i = 0; i < nev; ++i)
				eigenvalues.push_back(eval(i, 0));
			break;
		}
	}
}