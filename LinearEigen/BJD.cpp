#include<BJD.h>

BJD::BJD(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int restart, int batch, int gmres_size) 
	: LinearEigenSolver(A, B, nev),
	restart(restart),
	gmres_size(gmres_size),
	nRestart(0),
	cgstep(cgstep),
	batch(batch),
	V(A.rows(), batch* restart),
	WA(A.rows(), batch* restart),
	H(batch* restart, batch* restart) {

	Map<MatrixXd> V1(&V(0, 0), A.rows(), batch);
	V1 = MatrixXd::Random(A.rows(), batch);
	orthogonalization(V1, B);
	
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
	H.block(0, 0, batch, batch) = W1.transpose() * V1;
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
			
			cout << "第" << nIter - 1 << "轮重启：" << endl;
			cout << "迭代步：" << i << endl;
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

			system("cls");
			int cnv = conv_select(eval, ui, 0, tmpeval, tmpevec);
			com_of_mul += (A.nonZeros() + B.nonZeros() + 3 * A.rows()) * LinearEigenSolver::CHECKNUM;
			cout << "已收敛特征向量个数：" << cnv << endl;

			//t1 = clock();
			//tCnv += t1 - t2;

			if (cnv > prev)
				break;

			time_t now = time(&now);
			if (timeCheck(start_time, now))
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

			orthogonalization(X, eigenvectors, B);
			com_of_mul += X.cols() * eigenvectors.cols() * A.rows() * 2;

			orthogonalization(X, Vj, B);
			com_of_mul += X.cols() * Vj.cols() * A.rows() * 2;

			int dep = orthogonalization(X, B);
			com_of_mul += (X.cols() + 1) * X.cols() * A.rows();

			//t1 = clock();
			//tOrt += t1 - t2;

			new (&X) Map<MatrixXd>(&V(0, Vj.cols()), A.rows(), ri.cols() - dep);
			Map<MatrixXd> tmpWA(&WA(0, Vj.cols()), A.rows(), X.cols());
			tmpWA = A * X;
			com_of_mul += A.nonZeros() * X.cols();

			//t2 = clock();
			//tAX += t2 - t1;

			/*Map<MatrixXd> tmpWB(&WB(0, Vj.cols()), A.rows(), X.cols());
			tmpWB = B * X;*/

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

		}
		if (eigenvalues.size() >= nev)
			break;
		
		int left = nd - (eigenvalues.size() - prev);
		nd = batch < A.rows() - eigenvalues.size() ? batch : A.rows() - eigenvalues.size();
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
