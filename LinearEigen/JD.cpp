#include<JD.h>

JD::JD(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int restart, int batch, int gmres_size)
	: LinearEigenSolver(A, B, nev), restart(restart), gmres_size(gmres_size), cgstep(cgstep), batch(batch), 
	U(A.rows(), nev + batch * restart), V(&U(0, 0), U.rows(), batch * restart) {

	Map<MatrixXd> V1(&V(0, 0), A.rows(), batch);
	V1 = MatrixXd::Random(A.rows(), batch);
	orthogonalization(V1, B);
	W.resize(A.rows(), batch * restart);
	Map<MatrixXd> W1(&W(0, 0), A.rows(), batch);
	W1 = A * V1;
	H.resize(batch * restart, batch * restart);
	H.block(0, 0, batch, batch) = V1.transpose() * W1;
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
		//Map<MatrixXd> ui(&U(0, 0), U.rows(), 0);
		for (i = 0; i < restart; ++i) {
			system("cls");
			cout << "第" << nIter - 1 << "轮重启：" << endl;
			cout << "迭代步：" << i + 1 << endl;
			eval.resize(Vj.cols(), 1);
			evec.resize(Vj.cols(), Vj.cols());
			cout << Vj.cols() << endl;
			RR(H.block(0, 0, Vj.cols(), Vj.cols()), eval, evec);
			
			MatrixXd tmpu = Vj * evec;
			
			vector<int> cnv = conv_check(eval, tmpu, 0);
			if (cnv.size()) {
				int new_cnv = 0;
				int new_ui = 0;
				int prev = eigenvalues.size();
				new (&V) Map<MatrixXd>(&U(0, prev + cnv.size()), U.rows(), batch * restart);
				new (&Vj) Map<MatrixXd>(&V(0, 0), V.rows(), batch - cnv.size());
				for (int j = 0; j < tmpu.cols(); ++j) {
					if (eigenvalues.size() == nev)
						break;
					if ((new_cnv < cnv.size()) && (j == cnv[new_cnv])) {
						eigenvalues.push_back(eval(j, 0));
						eigenvectors.col(prev + new_cnv) = tmpu.col(j);
						U.col(prev + new_cnv) = tmpu.col(j);
						++new_cnv;
					}
					else {
						Vj.col(new_ui) = tmpu.col(j);
						++new_ui;
					}
				}
				orthogonalization(Vj, Map<MatrixXd>(&eigenvectors(0, 0), A.rows(), eigenvalues.size()), B);
				orthogonalization(Vj, B);
				break;
			}
			else if (i == restart - 1) {
				new (&Vj) Map<MatrixXd>(&V(0, 0), V.rows(), batch);
				for (int j = 0; j < tmpu.cols(); ++j)
					Vj.col(j) = tmpu.col(j);
				break;
			}

			MatrixXd ri = MatrixXd::Zero(tmpu.rows(), batch);
			for (int j = 0; j < ri.cols(); ++j) {
				ri.col(j) += B * tmpu.col(j) * eval(j, 0);
			}
			ri -= A * tmpu.leftCols(batch);
			
			Map<MatrixXd> X(&V(0, Vj.cols()), A.rows(), ri.cols());
			X = MatrixXd::Zero(A.rows(), ri.cols());
			
			Map<MatrixXd> ui(&U(0, 0), U.rows(), eigenvalues.size() + Vj.cols());
			L_GMRES(A, B, ri, tmpu, X, eval, gmres_size);
			//cout << X << endl << endl;
			orthogonalization(X, ui, B);
			//cout << X << endl << endl;
			//cout << Vj << endl << endl;
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
		Map<MatrixXd> tmpW(&W(0, Vj.cols()), A.rows(), Vj.cols());
		tmpW = A * Vj;
		H.block(0, 0, Vj.cols(), Vj.cols()) = Vj.transpose() * tmpW;
	}
}
