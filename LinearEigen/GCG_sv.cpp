#include<GCG_sv.h>
#include<iostream>

using namespace std;

GCG_sv::GCG_sv(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep) : GCG_sv(A, B, nev, cgstep, nev){}
GCG_sv::GCG_sv(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int batch) : LinearEigenSolver(A, B, nev){
	this->batch = batch;
	X = MatrixXd::Random(A.rows(), nev);
	orthogonalization(X, B);
	MatrixXd eval, evec;
	projection_RR(X, A, eval, evec);
	X = X * evec;
	this->eval = eval;
	cout << "X初始化" << endl;
	LAM.resize(batch, batch);
	LAM.reserve(batch);
	for (int i = 0; i < batch; ++i) {
		LAM.insert(i, i) = eval.col(0)[i];
	}
	cout << "λ初始化" << endl;
	//linearsolver.compute(A);
	linearsolver.setMaxIterations(cgstep);
	cout << "CG求解器准备完成..." << endl;
	P.resize(A.rows(), 0);
	X0.resize(A.rows(), 0);
	eigenvectors.resize(A.rows(), nev);
	nIter = 0;
	cout << "初始化完成" << endl;
}
void GCG_sv::compute() {

	double shift = 0;
	int cnv = 0;
	MatrixXd eval, evec, Xnew, tmp, X1;
	SparseMatrix<double> tmpA;
	MatrixXd tmpAX1;
	MatrixXd ty;
	W2.resize(A.rows(), 0);
	while (true) {
		++nIter;
		cout << "迭代步：" << nIter << endl;
		cout << "移频：" << shift << endl;
		X1 = X.block(0, 0, A.rows(), batch);
		tmpA = A + shift * B;
		linearsolver.compute(tmpA);
		W1 = linearsolver.solveWithGuess(B * X1 * LAM, X1);
		/*if (W2.cols()) {
			tmpAX1 = tmpA * W1;
			linearsolver.compute(LAM);
			ty = linearsolver.solve(tmpAX1.transpose());
			linearsolver.compute(B);
			W2 = linearsolver.solve(ty.transpose());
		}*/
		tmp = tmpA * W1 - B * X1 * LAM;
		cout << "W1求解残差" << tmp.col(0).norm() << endl;
		if (nIter > 1) {
			/*if (W2.cols())
				V.resize(A.rows(), X.cols() + LAM.cols() + P.cols() + W2.cols());
			else*/
			V.resize(A.rows(), X.cols() + LAM.cols() + P.cols());
		}
		else {
			V.resize(A.rows(), X.cols() + LAM.cols());
		}

		/*if (W2.cols())
			V << W1, W2, X, P;
		else*/
		V << W1, X, P;
		orthogonalization(V, X0, B);
		orthogonalization(V, B);
		cout << V.cols() << endl;

		projection_RR(V, tmpA, eval, evec);

		cout << V.rows() << " " << V.cols() << " " << evec.rows() << " " << evec.cols() << endl;
		Xnew = V * evec;
		//TODO
		//cnv = conv_check(A, B, eval, Xnew, shift);
		system("cls");
		cout << "已收敛特征向量个数：" << cnv + X0.cols() << endl;
		if (cnv) {
			int need = min((int)(nev - X0.cols()), cnv);
			for (int i = 0; i < need; ++i) {
				eigenvalues.push_back(eval.col(0)[i] - shift);
				eigenvectors.col(X0.cols() + i) = Xnew.col(i);
				cout << "存储特征值：" << eval.col(0)[i] - shift << "，移频" << shift << endl;
				//cout << "存储特征向量：" << eigenvectors.col(X0.cols() + i).transpose() << endl;
			}
			if (eigenvalues.size() >= nev)
				break;
			tmp.resize(X0.rows(), X0.cols() + cnv);
			tmp << X0, Xnew.block(0, 0, A.rows(), cnv);
			X0 = tmp;
			W2.resize(A.rows(), 1);
		}
		else
			W2.resize(A.rows(), 0);
		LAM.resize(batch, batch);
		LAM.reserve(batch);
		for (int i = cnv; i < cnv + batch; ++i) {
			LAM.insert(i - cnv, i - cnv) = eval.col(0)[i] - shift;
		}
		P = X1;
		X = Xnew.block(0, cnv, A.rows(), max((int)(nev - X0.cols()), batch));

		//shift = ((eval.col(0)[cnv + batch - 1] - shift) - 100 * (eval.col(0)[cnv] - shift)) / 99;
		/*if (eigenvalues.size())
			shift = ((eval.col(0)[nev - X0.cols() + cnv - 1] - shift) - 100 * (eigenvalues[0])) / 99;
		else
			shift = ((eval.col(0)[nev - 1] - shift) - 100 * (eval.col(0)[0] - shift)) / 99;*/
		for (int i = 0; i < batch; ++i) {
			LAM.coeffRef(i, i) += shift;
		}
	}
}