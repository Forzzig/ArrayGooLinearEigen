#include<LOBPCG_I_Batch.h>

using namespace std;
LOBPCG_I_Batch::LOBPCG_I_Batch(SparseMatrix<double, RowMajor>& A, SparseMatrix<double, RowMajor>& B, int nev, int cgstep, int batch)
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
	int dep = orthogonalization(X, B);
	while (dep) {
		X.rightCols(dep) = MatrixXd::Random(A.rows(), dep);
		dep = orthogonalization(X, B);
	}
	cout << "X初始化" << endl;

	//TODO 只有要求解的矩阵不变时才能在这初始化
	linearsolver.compute(A);

	//事先固定CG迭代步数量
	linearsolver.setMaxIterations(cgstep);
	cout << "CG求解器准备完成..." << endl;
	cout << "初始化完成" << endl;
}
LOBPCG_I_Batch::~LOBPCG_I_Batch() {
	delete[] storage;
}

void LOBPCG_I_Batch::compute() {
	MatrixXd eval(batch, 1), evec(batch, batch), AX(A.rows(), batch), BX(A.rows(), batch), AWP(A.rows(), batch), Xnew(A.rows(), 2 * batch);

	//所有已计算出来的特征向量和待计算的都依次存在storage里，避免内存拷贝
	Map<MatrixXd> V(storage, A.rows(), X.cols() + W.cols());
	Map<MatrixXd> WP(storage + A.rows() * X.cols(), A.rows(), W.cols());
	while (true) {
		++nIter;
		cout << "迭代步：" << nIter << endl;

		if (AX.cols() != X.cols()) 
			AX.resize(NoChange, X.cols());
		AX.noalias() = A * X;
		com_of_mul += A.nonZeros() * X.cols();

		if (BX.cols() != X.cols())
			BX.resize(NoChange, X.cols());
		BX.noalias() = B * X;
		com_of_mul += B.nonZeros() * X.cols();
		
		H.block(0, 0, X.cols(), X.cols()).noalias() = X.transpose() * AX;
		com_of_mul += X.cols() * A.rows() * X.cols();

#pragma omp parallel for
		for (int i = 0; i < X.cols(); ++i) {
			BX.col(i) *= H(i, i);
		}
		com_of_mul += A.rows() * X.cols();

		//求解 A*W = A*X - mu*B*X， X为上一步的近似特征向量
		W = linearsolver.solve(AX - BX);
		com_of_mul += X.cols() * (A.nonZeros() + 4 * A.rows() +
			cgstep * (A.nonZeros() + 7 * A.rows()));

		//预存上一步的近似特征向量，BX废物利用
		memcpy(&BX(0, 0), &X(0, 0), X.rows() * X.cols() * sizeof(double));
		//BX = X;

		if ((WP.cols() != W.cols() + P.cols()) || (&WP(0, 0) != storage + A.rows() * X.cols()))
			new (&WP) Map<MatrixXd>(storage + A.rows() * X.cols(), A.rows(), W.cols() + P.cols());
		orthogonalization(WP, eigenvectors, B);
		orthogonalization(WP, X, B);

		int dep = orthogonalization(WP, B);

		//正交化后会有线性相关项，剔除
		if (dep)
			new (&WP) Map<MatrixXd>(storage + A.rows() * X.cols(), A.rows(), W.cols() + P.cols() - dep);

		if (AWP.cols() != WP.cols())
			AWP.resize(NoChange, WP.cols());
		AWP.noalias() = A * WP;
		com_of_mul += A.nonZeros() * WP.cols();

		H.block(0, X.cols(), X.cols(), WP.cols()).noalias() = X.transpose() * AWP;
		H.block(X.cols(), 0, WP.cols(), X.cols()) = H.block(0, X.cols(), X.cols(), WP.cols()).transpose();
		com_of_mul += X.cols() * A.rows() * WP.cols();

		H.block(X.cols(), X.cols(), WP.cols(), WP.cols()).noalias() = WP.transpose() * AWP;
		com_of_mul += WP.cols() * A.rows() * WP.cols();

		if (V.cols() != X.cols() + WP.cols()) 
			new (&V) Map<MatrixXd>(storage, A.rows(), X.cols() + WP.cols());

		RR(H.block(0, 0, V.cols(), V.cols()), eval, evec);

		//子空间V投影下的新的近似特征向量
		int nd = (2 * batch < V.cols()) ? 2 * batch : V.cols();
		Xnew.leftCols(nd) = V * evec.leftCols(nd);
		com_of_mul += A.rows() * V.cols() * nd;

		system("cls");
		int prev = eigenvalues.size();
		int cnv = conv_select(eval, Xnew, 0, eval, X);
		cout << "LOBPCG-I已收敛特征向量个数：" << cnv << endl;

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
		dep = orthogonalization(X, B);
		while (dep) {
			X.rightCols(dep) = MatrixXd::Random(A.rows(), dep);
			dep = orthogonalization(X, B);
		}

		if ((P.cols() != BX.cols()) || (&P(0, 0) != storage + A.rows() * (X.cols() + W.cols()))) 
			new (&P) Map<MatrixXd>(storage + A.rows() * (X.cols() + W.cols()), A.rows(), BX.cols());
		memcpy(&P(0, 0), &BX(0, 0), A.rows() * BX.cols() * sizeof(double));
		//P = BX;
	}
	finish();
}