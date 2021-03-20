#include<LOBPCG_I_Batch.h>

using namespace std;
LOBPCG_I_Batch::LOBPCG_I_Batch(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int batch)
	: LinearEigenSolver(A, B, nev),
	//摆放顺序：XWP
	storage(new double[A.rows() * batch * 3]),
	X(storage, A.rows(), batch),
	W(storage + A.rows() * batch, A.rows(), batch),
	P(storage + A.rows() * batch * 2, A.rows(), 0),

	H(batch * 3, batch * 3),
	batch(batch),
	cgstep(cgstep){

	X = MatrixXd::Random(A.rows(), batch);
	orthogonalization(X, B);
	cout << "X初始化" << endl;

	//TODO 只有要求解的矩阵不变时才能在这初始化
	linearsolver.compute(A);

	//事先固定CG迭代步数量
	linearsolver.setMaxIterations(cgstep);
	cout << "CG求解器准备完成..." << endl;
	cout << "初始化完成" << endl;
}


void LOBPCG_I_Batch::compute() {
	MatrixXd eval, evec, tmp, Xnew, AX, BX, AWP;

	//所有已计算出来的特征向量和待计算的都依次存在storage里，避免内存拷贝
	Map<MatrixXd> V(storage, A.rows(), X.cols() + W.cols());
	Map<MatrixXd> WP(storage + A.rows() * X.cols(), A.rows(), W.cols());
	while (true) {
		++nIter;
		cout << "迭代步：" << nIter << endl;
		AX = A * X;
		com_of_mul += A.nonZeros() * X.cols();

		BX = B * X;
		com_of_mul += B.nonZeros() * X.cols();
		
		H.block(0, 0, X.cols(), X.cols()) = X.transpose() * AX;
		com_of_mul += X.cols() * A.rows() * X.cols();

		tmp = BX;
		for (int i = 0; i < X.cols(); ++i) {
			tmp.col(i) *= H(i, i);
		}
		com_of_mul += A.rows() * X.cols();

		//求解 A*W = A*X - mu*B*X， X为上一步的近似特征向量
		W = linearsolver.solve(AX - tmp);
		com_of_mul += X.cols() * (A.nonZeros() + 4 * A.rows() +
			cgstep * (A.nonZeros() + 7 * A.rows()));

		//预存上一步的近似特征向量
		tmp = X;

		new (&WP) Map<MatrixXd>(storage + A.rows() * X.cols(), A.rows(), W.cols() + P.cols());
		orthogonalization(WP, eigenvectors, B);
		orthogonalization(WP, X, B);

		int dep = orthogonalization(WP, B);

		//正交化后会有线性相关项，剔除
		new (&WP) Map<MatrixXd>(storage + A.rows() * X.cols(), A.rows(), W.cols() + P.cols() - dep);
		AWP = A * WP;
		com_of_mul += A.nonZeros() * WP.cols();

		H.block(0, X.cols(), X.cols(), WP.cols()) = X.transpose() * AWP;
		H.block(X.cols(), 0, WP.cols(), X.cols()) = H.block(0, X.cols(), X.cols(), WP.cols()).transpose();
		com_of_mul += X.cols() * A.rows() * WP.cols();

		H.block(X.cols(), X.cols(), WP.cols(), WP.cols()) = WP.transpose() * AWP;
		com_of_mul += WP.cols() * A.rows() * WP.cols();

		new (&V) Map<MatrixXd>(storage, A.rows(), X.cols() + WP.cols());

		RR(H.block(0, 0, V.cols(), V.cols()), eval, evec);

		//子空间V投影下的新的近似特征向量
		int nd = (2 * batch < V.cols()) ? 2 * batch : V.cols();
		Xnew = V * evec.block(0, 0, V.cols(), nd);
		com_of_mul += A.rows() * V.cols() * nd;

		system("cls");
		int prev = eigenvalues.size();
		int cnv = conv_select(eval, Xnew, 0, eval, X);
		cout << "已收敛特征向量个数：" << cnv << endl;

		if (cnv >= nev)
			break;

		time_t now = time(&now);
		if (timeCheck(start_time, now))
			break;

		if (nd - (cnv - prev) < X.cols()) {
			int wid = nd - (cnv - prev);
			new (&X) Map<MatrixXd>(storage, A.rows(), wid);
			new (&W) Map<MatrixXd>(storage + A.rows() * X.cols(), A.rows(), wid);
		}
		orthogonalization(X, eigenvectors, B);
		orthogonalization(X, B);

		new (&P) Map<MatrixXd>(storage + A.rows() * (X.cols() + W.cols()), A.rows(), tmp.cols());
		P = tmp;
	}
	finish();
}