#include<JD.h>

JD::JD(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int restart, int batch, int gmres_size) : LinearEigenSolver(A, B, nev) {
	this->restart = restart;
	this->gmres_size = gmres_size;
	V.resize(A.rows(), batch * restart);
	Map<MatrixXd> V1(&V(0, 0), A.rows(), batch);
	V1 = MatrixXd::Random(A.rows(), batch);
	orthogonalization(V1, B);
	W.resize(A.rows(), batch * restart);
	Map<MatrixXd> W1(&W(0, 0), A.rows(), batch);
	W1 = A * V1;
	H.resize(batch * restart, batch * restart);
	H.block(0, 0, batch, batch) = V1.transpose() * W1;
	this->cgstep = cgstep;
	this->batch = batch;
	eigenvectors.resize(A.rows(), 0);

	//cout << scientific << setprecision(32);
	////A.resize(20, 20);
	////B.resize(20, 20);
	////for (int i = 0; i < A.rows(); ++i) {
	////	A.insert(i, i) = rand() + 1;
	////	B.insert(i, i) = 1;
	////	for (int j = i + 1; j < A.rows(); ++j) {
	////		A.insert(i, j) = rand() + 1;
	////		A.insert(j, i) = A.coeff(i, j);
	////	}
	////}
	//MatrixXd b = MatrixXd::Random(A.rows(), 1);
	//MatrixXd U = MatrixXd::Random(A.rows(), 2);
	//MatrixXd X = MatrixXd::Zero(A.rows(), 1);
	//MatrixXd lam(1, 1);
	//lam(0, 0) = 0;

	//cout << A << endl;
	//cout << "-----------------------" << endl;
	//cout << X << endl;
	//cout << "-----------------------" << endl;
	//cout << b << endl;
	//cout << "-----------------------" << endl;
	//cout << U << endl;
	//system("pause");
	//specialCG_Bad(A, B, b, U, X, lam);
}

void JD::compute() {
	MatrixXd eval, evec;
	Map<MatrixXd> Vj(&V(0, 0), A.rows(), batch);
	while (true) {
		++nIter;
		int i;
		MatrixXd tmpeval, tmpevec;
		for (i = 1; i <= restart; ++i) {
			system("cls");
			cout << "第" << nIter - 1 << "轮重启：" << endl;
			cout << "迭代步：" << i << endl;
			eval.resize(Vj.cols(), 1);
			evec.resize(Vj.cols(), Vj.cols());
			cout << Vj.cols() << endl;
			RR(H.block(0, 0, Vj.cols(), Vj.cols()), eval, evec);
			//注意eval长度是偏长的
			MatrixXd ui = Vj * evec.block(0, 0, Vj.cols(), batch);
			MatrixXd ri = MatrixXd::Zero(ui.rows(), ui.cols());
			for (int j = 0; j < ri.cols(); ++j) {
				ri.col(j) += B * ui.col(j) * eval(j, 0);
			}
			ri -= A * ui;
			int prev = eigenvalues.size();
			tmpeval.resize(batch, 1);
			tmpevec.resize(A.rows(), batch);
			int cnv = conv_select(eval, ui, 0, tmpeval, tmpevec);
			if (cnv > prev)
				break;
			if (i == restart)
				break;

			Map<MatrixXd> X(&V(0, Vj.cols()), A.rows(), ri.cols());
			X = MatrixXd::Zero(A.rows(), ri.cols());
			
			L_GMRES(A, B, ri, ui, X, eval, gmres_size);
			//cout << X << endl << endl;
			orthogonalization(X, eigenvectors, B);
			//cout << X << endl << endl;
			//cout << Vj << endl << endl;
			orthogonalization(X, Vj, B);
			//cout << X << endl << endl;
			int dep = orthogonalization(X, B);
			//cout << X << endl << endl;
			
			new (&X) Map<MatrixXd>(&V(0, Vj.cols()), A.rows(), ri.cols() - dep);
			/*cout << X << endl<< endl;*/
			Map<MatrixXd> tmpW(&W(0, Vj.cols()), A.rows(), X.cols());
			tmpW = A * X;
			/*cout << tmpW << endl;*/
			H.block(Vj.cols(), 0, X.cols(), Vj.cols()) = tmpW.transpose() * Vj;
			H.block(0, Vj.cols(), Vj.cols(), X.cols()) = H.block(Vj.cols(), 0, X.cols(), Vj.cols()).transpose();
			H.block(Vj.cols(), Vj.cols(), X.cols(), X.cols()) = tmpW.transpose() * X;
			new (&Vj) Map<MatrixXd>(&V(0, 0), A.rows(), Vj.cols() + X.cols());
			
			/*cout << Vj << endl;
			cout << Vj.transpose() * B * Vj << endl;
			cout << Vj.transpose() * A * Vj << endl;
			cout << H.block(0, 0, Vj.cols(), Vj.cols()) << endl;*/
		}
		if (eigenvalues.size() >= nev)
			break;
		new (&Vj) Map<MatrixXd>(&V(0, 0), A.rows(), batch);
		Vj = tmpevec.block(0, 0, A.rows(), batch);
		if (eigenvalues.size() > 0)
			orthogonalization(Vj, Map<MatrixXd>(&eigenvectors(0, 0), A.rows(), eigenvalues.size()), B);
		orthogonalization(Vj, B);
		Map<MatrixXd> tmpW(&W(0, Vj.cols()), A.rows(), Vj.cols());
		tmpW = A * Vj;
		H.block(0, 0, Vj.cols(), Vj.cols()) = Vj.transpose() * tmpW;
	}
}
