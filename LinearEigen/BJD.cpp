#include<BJD.h>
#include<ctime>

BJD::BJD(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int restart, int batch, int gmres_size) : LinearEigenSolver(A, B, nev) {
	this->restart = restart;
	this->gmres_size = gmres_size;
	V.resize(A.rows(), batch * restart);
	Map<MatrixXd> V1(&V(0, 0), A.rows(), batch);
	V1 = MatrixXd::Random(A.rows(), batch);
	orthogonalization(V1, B);
	WA.resize(A.rows(), batch * restart);
	Map<MatrixXd> W1(&WA(0, 0), A.rows(), batch);
	W1 = A * V1;
	/*WB.resize(A.rows(), batch * restart);
	Map<MatrixXd> W2(&WB(0, 0), A.rows(), batch);
	W2 = B * V1;
	HA.resize(batch * restart, batch * restart);
	HA.block(0, 0, batch, batch) = W1.transpose() * W1;
	HB.resize(batch * restart, batch * restart);
	HB.block(0, 0, batch, batch) = W2.transpose() * W2;
	HAB.resize(batch * restart, batch * restart);
	HAB.block(0, 0, batch, batch) = W1.transpose() * W2;*/
	H.resize(batch * restart, batch * restart);
	H.block(0, 0, batch, batch) = W1.transpose() * V1;
	this->cgstep = cgstep;
	this->batch = batch;
	eigenvectors.resize(A.rows(), 0);

}

void BJD::compute() {
	MatrixXd eval, evec, ui, ri;
	Map<MatrixXd> Vj(&V(0, 0), A.rows(), batch);
	Map<MatrixXd> WAj(&WA(0, 0), A.rows(), batch);
	/*Map<MatrixXd> WBj(&WB(0, 0), A.rows(), batch);*/

	MatrixXd tmpeval(batch, 1), tmpevec(A.rows(), batch);
	int nd = batch;
	/*long long t1, t2;
	long long tRR = 0, tCnv = 0, tGMR = 0, tOrt = 0, tAX = 0, tH = 0;*/
	while (true) {
		++nIter;
		int prev = eigenvalues.size();
		for (int i = 1; i <= restart; ++i) {
			
			system("cls");
			cout << "第" << nIter - 1 << "轮重启：" << endl;
			cout << "迭代步：" << i << endl;
			cout << Vj.cols() << endl;

			//t1 = clock();

			//generalized_RR(HA.block(0, 0, Vj.cols(), Vj.cols()), HB.block(0, 0, Vj.cols(), Vj.cols()), HAB.block(0, 0, Vj.cols(), Vj.cols()), 0, eval, evec);
			RR(H.block(0, 0, Vj.cols(), Vj.cols()), eval, evec);
			com_of_mul += 4 * Vj.cols() * Vj.cols() * Vj.cols();
			
			//t2 = clock();
			//tRR += t2 - t1;

			//注意eval长度是偏长的
			ui = Vj * evec.leftCols(nd);
			com_of_mul += A.rows() * Vj.cols() * evec.cols();

			ri = B * ui;
			for (int j = 0; j < ri.cols(); ++j) {
				ri.col(j) *= eval(j, 0);
			}
			com_of_mul += ui.cols() * (B.nonZeros() + A.rows());

			ri -= A * ui;
			com_of_mul += A.nonZeros() * ui.cols();
			
			/*coutput << "V--------------------------------" << endl << Vj << endl;
			coutput << "WA--------------------------------" << endl << WAj << endl;
			coutput << "WB--------------------------------" << endl << WBj << endl;
			coutput << "HA--------------------------------" << endl << HA.block(0, 0, Vj.cols(), Vj.cols()) << endl;
			coutput << "HB--------------------------------" << endl << HB.block(0, 0, Vj.cols(), Vj.cols()) << endl;
			coutput << "HAB--------------------------------" << endl << HAB.block(0, 0, Vj.cols(), Vj.cols()) << endl;
			coutput << "H--------------------------------" << endl << H.block(0, 0, Vj.cols(), Vj.cols()) << endl;
			coutput << "eval--------------------------------" << endl << eval << endl;
			coutput << "evec--------------------------------" << endl << evec << endl;
			coutput << "ui--------------------------------" << endl << ui << endl;*/

			int cnv = conv_select(eval, ui, 0, tmpeval, tmpevec);
			com_of_mul += (A.nonZeros() + B.nonZeros() + 3 * A.rows()) * LinearEigenSolver::CHECKNUM;

			//t1 = clock();
			//tCnv += t1 - t2;

			if (cnv > prev)
				break;
			if (i == restart)
				break;

			Map<MatrixXd> X(&V(0, Vj.cols()), A.rows(), ri.cols());
			X = MatrixXd::Zero(A.rows(), ri.cols());

			L_GMRES(A, B, ri, ui, X, eval, gmres_size);
			com_of_mul += A.rows() + A.rows() * ui.cols() + ui.cols() * A.rows() * ui.cols() + ui.cols() * ui.cols() * ui.cols() +
				ri.cols() * (A.nonZeros() + ui.rows() * ui.cols() + ui.cols() * ui.rows()
							+ A.rows() + ui.cols() * ui.rows() + ui.cols() * ui.cols() + ui.rows() * ui.cols()
							+ A.rows() + ui.cols() * ui.rows() + ui.cols() * ui.cols() + ui.rows() * ui.cols()
							+ cgstep * (A.nonZeros() + ui.rows() * ui.cols() + ui.cols() * ui.rows()
											+ A.rows() + ui.cols() * ui.rows() + ui.cols() * ui.cols() + ui.rows() * ui.cols()
											+ gmres_size * (gmres_size + 1) / 2 * (A.rows() + ui.cols())
											+ A.rows() + ui.cols())
							+ cgstep * (A.rows() + ui.cols()));

			//t2 = clock();
			//tGMR += t2 - t1;

			//cout << X << endl << endl;
			orthogonalization(X, eigenvectors, B);
			com_of_mul += X.cols() * eigenvectors.cols() * A.rows() * 2;

			//cout << X << endl << endl;
			//cout << Vj << endl << endl;
			orthogonalization(X, Vj, B);
			com_of_mul += X.cols() * Vj.cols() * A.rows() * 2;

			//cout << X << endl << endl;
			int dep = orthogonalization(X, B);
			com_of_mul += (X.cols() + 1) * X.cols() * A.rows();

			//cout << X << endl << endl;
			//t1 = clock();
			//tOrt += t1 - t2;

			new (&X) Map<MatrixXd>(&V(0, Vj.cols()), A.rows(), ri.cols() - dep);
			/*cout << X << endl<< endl;*/
			Map<MatrixXd> tmpWA(&WA(0, Vj.cols()), A.rows(), X.cols());
			tmpWA = A * X;
			com_of_mul += A.nonZeros() * X.cols();

			//t2 = clock();
			//tAX += t2 - t1;

			/*Map<MatrixXd> tmpWB(&WB(0, Vj.cols()), A.rows(), X.cols());
			tmpWB = B * X;*/

			/*cout << tmpW << endl;*/

			/*HA.block(Vj.cols(), 0, X.cols(), Vj.cols()) = tmpWA.transpose() * WAj;
			HA.block(0, Vj.cols(), Vj.cols(), X.cols()) = HA.block(Vj.cols(), 0, X.cols(), Vj.cols()).transpose();
			HA.block(Vj.cols(), Vj.cols(), X.cols(), X.cols()) = tmpWA.transpose() * tmpWA;
			HB.block(Vj.cols(), 0, X.cols(), Vj.cols()) = tmpWB.transpose() * WBj;
			HB.block(0, Vj.cols(), Vj.cols(), X.cols()) = HB.block(Vj.cols(), 0, X.cols(), Vj.cols()).transpose();
			HB.block(Vj.cols(), Vj.cols(), X.cols(), X.cols()) = tmpWB.transpose() * tmpWB;
			HAB.block(Vj.cols(), 0, X.cols(), Vj.cols()) = tmpWA.transpose() * WBj;
			HAB.block(0, Vj.cols(), Vj.cols(), X.cols()) = HAB.block(Vj.cols(), 0, X.cols(), Vj.cols()).transpose();
			HAB.block(Vj.cols(), Vj.cols(), X.cols(), X.cols()) = tmpWA.transpose() * tmpWB;*/
			H.block(Vj.cols(), 0, X.cols(), Vj.cols()) = tmpWA.transpose() * Vj;
			H.block(0, Vj.cols(), Vj.cols(), X.cols()) = H.block(Vj.cols(), 0, X.cols(), Vj.cols()).transpose();
			H.block(Vj.cols(), Vj.cols(), X.cols(), X.cols()) = tmpWA.transpose() * X;
			com_of_mul += X.cols() * A.rows() * Vj.cols() + X.cols() * A.rows() * X.cols();

			//t1 = clock();
			//tH += t1 - t2;

			new (&Vj) Map<MatrixXd>(&V(0, 0), A.rows(), Vj.cols() + X.cols());
			new (&WAj) Map<MatrixXd>(&WA(0, 0), A.rows(), WAj.cols() + X.cols());
			/*new (&WBj) Map<MatrixXd>(&WB(0, 0), A.rows(), WBj.cols() + X.cols());*/

			/*cout << Vj << endl;
			cout << Vj.transpose() * B * Vj << endl;
			cout << Vj.transpose() * A * Vj << endl;
			cout << H.block(0, 0, Vj.cols(), Vj.cols()) << endl;*/
		}
		if (eigenvalues.size() >= nev)
			break;
		
		int left = nd - (eigenvalues.size() - prev);
		nd = batch < nev - eigenvalues.size() ? batch : nev - eigenvalues.size();
		new (&Vj) Map<MatrixXd>(&V(0, 0), A.rows(), nd);
		Vj.leftCols(left) = tmpevec.leftCols(left);
		Vj.rightCols(nd - left) = MatrixXd::Random(A.rows(), nd - left);
		if (eigenvalues.size() > 0)
			orthogonalization(Vj, Map<MatrixXd>(&eigenvectors(0, 0), A.rows(), eigenvalues.size()), B);
		orthogonalization(Vj, B);
		
		new (&WAj) Map<MatrixXd>(&WA(0, 0), A.rows(), Vj.cols());
		/*new (&WBj) Map<MatrixXd>(&WB(0, 0), A.rows(), Vj.cols());*/

		WAj = A * Vj;
		com_of_mul += A.nonZeros() * Vj.cols();

		/*WBj = B * Vj;
		HA.block(0, 0, Vj.cols(), Vj.cols()) = WAj.transpose() * WAj;
		HB.block(0, 0, Vj.cols(), Vj.cols()) = WBj.transpose() * WBj;
		HAB.block(0, 0, Vj.cols(), Vj.cols()) = WAj.transpose() * WBj;*/
		H.block(0, 0, Vj.cols(), Vj.cols()) = WAj.transpose() * Vj;
		com_of_mul += Vj.cols() * A.rows() * Vj.cols();
	}
	/*cout << "RR   , Cnv   , GMR   , Ort   , AX   , H" << endl;
	cout << tRR << ", " << tCnv << ", " << tGMR << ", " << tOrt << ", " << tAX << ", " << tH << endl;
	system("pause");*/
}
