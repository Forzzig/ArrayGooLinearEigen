#include<LOBPCG_II.h>
#include<iostream>

using namespace std;
LOBPCG_II::LOBPCG_II(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep)
	: LinearEigenSolver(A, B, nev),
	storage(new double[A.rows() * nev * 3]),
	X(storage + A.rows(), A.rows(), nev, OuterStride<>(3 * A.rows())),
	W(storage, A.rows(), nev, OuterStride<>(3 * A.rows())),
	P(storage + 2 * A.rows(), A.rows(), 0, OuterStride<>(3 * A.rows())) {
	
	X = MatrixXd::Random(A.rows(), nev);
	orthogonalization(X, B);
	cout << "X初始化" << endl;

	//只有要求解的矩阵不变时才能在这初始化
	linearsolver.compute(A);

	//事先固定CG迭代步数量
	this->cgstep = cgstep;
	linearsolver.setMaxIterations(cgstep);
	cout << "CG求解器准备完成..." << endl;
	cout << "初始化完成" << endl;
}


void LOBPCG_II::compute() {
	vector<int> cnv;
	MatrixXd eval, evec, tmp, tmpA, mu;
	Map<MatrixXd> V(storage, A.rows(), 3), v1(storage, 0, 0);
	while (true) {
		++nIter;
		cout << "迭代步：" << nIter << endl;

		tmpA = A * X;
		com_of_mul += A.nonZeros() * X.cols();

		tmp = B * X;
		com_of_mul += B.nonZeros() * X.cols();

		for (int i = 0; i < nev; ++i) {
			mu = X.col(i).transpose() * tmpA.col(i);
			tmp.col(i) *= mu(0, 0);
		}
		com_of_mul += (X.cols() + 1) * A.rows() * X.cols();

		//求解 A*W = A*X - mu*B*X， X为上一步的近似特征向量
		W = linearsolver.solve(tmpA - tmp);
		com_of_mul += X.cols() * (A.nonZeros() + 4 * A.rows() +
			cgstep * (A.nonZeros() + 7 * A.rows()));

		/*coutput << "W---------------------------------" << endl << W << endl;
		coutput << "WXP--------------------------------" << endl << Map<MatrixXd>(storage, A.rows(), nev * 3) << endl;*/

		//预存上一步的近似特征向量
		tmp = X;
		/*cout << W.cols() << " " << X.cols() << " " << P.cols()<< endl;*/

		//对每组XPW分别求解RR
		for (int i = 0; i < nev; ++i) {
			if (nIter == 1)
				new (&V) Map<MatrixXd>(storage + A.rows() * 3 * i, A.rows(), 2);
			else
				new (&V) Map<MatrixXd>(storage + A.rows() * 3 * i, A.rows(), 3);
			
			/*coutput << "nIter: " << nIter << "---------------------------" << endl;
			coutput << "i: " << i << "-----------------------------" << endl;
			coutput << "V--------------------------------" << endl << V << endl;*/

			int dep = orthogonalization(V, B);
			com_of_mul += (V.cols() + 1) * V.cols() * A.rows();

			int Vold = V.cols();
			new (&V) Map<MatrixXd>(storage + A.rows() * 3 * i, A.rows(), Vold - dep);
			
			projection_RR(V, A, eval, evec);
			com_of_mul += V.cols() * A.nonZeros() * V.cols() + (24 * V.cols() * V.cols() * V.cols());

			X.col(i) = V * evec.block(0, 0, V.cols(), 1);
			com_of_mul += A.rows() * V.cols();
		}
		
		new (&P) Map<MatrixXd, Unaligned, OuterStride<>>(storage + 2 * A.rows(), A.rows(), 0, OuterStride<>(3 * A.rows()));
		P = tmp;

		////全体做正交化，如果分块容易出问题
		//int dep = orthogonalization(V, B);
		//com_of_mul += (V.cols() + 1) * V.cols() * A.rows();

		////正交化后会有线性相关项，剔除
		//new (&V) Map<MatrixXd>(storage, A.rows(), W.cols() + X.cols() + P.cols() - dep);
		//cout << V.cols() << endl;

		//projection_RR(V, A, eval, evec);
		//com_of_mul += V.cols() * A.nonZeros() * V.cols() + (24 * V.cols() * V.cols() * V.cols());

		//system("cls");
		//cout << V.rows() << " " << V.cols() << " " << evec.rows() << " " << evec.cols() << endl;

		////子空间V投影下的新的近似特征向量
		//X = V * evec.block(0, 0, V.cols(), nev);
		//com_of_mul += A.rows() * V.cols() * nev;

		////P是上一步的近似特征向量
		//new (&P) Map<MatrixXd>(storage + A.rows() * (W.cols() + X.cols()), A.rows(), nev);
		//P = tmp;

		//对X再做一次RR问题
		orthogonalization(X, B); //TODO 这里应该不会有线性相关
		com_of_mul += (X.cols() + 1) * X.cols() * A.rows();

		projection_RR(X, A, eval, evec);
		com_of_mul += X.cols() * A.nonZeros() * X.cols() + (24 * X.cols() * X.cols() * X.cols());
		
		X *= evec;
		com_of_mul += A.rows() * X.cols() * X.cols();

		system("cls");
		cnv = conv_check(eval, X, 0);
		cout << "已收敛特征向量个数：" << cnv.size() << endl;
		com_of_mul += (A.nonZeros() + B.nonZeros() + 3 * A.rows()) * LinearEigenSolver::CHECKNUM;

		if (cnv.size() >= nev) {
			eigenvectors = X;
			for (int i = 0; i < nev; ++i)
				eigenvalues.push_back(eval(i, 0));
			break;
		}
	}
}