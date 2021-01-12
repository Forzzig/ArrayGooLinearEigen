#include<JD.h>

JD::JD(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int restart, int batch) : LinearEigenSolver(A, B, nev) {
	this->restart = restart;
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
			MatrixXd ri = A * ui;
			for (int j = 0; j < ri.cols(); ++j) {
				ri.col(j) -= B * ui.col(j) * eval(j, 0);
			}
			int prev = eigenvalues.size();
			tmpeval.resize(batch, 1);
			tmpevec.resize(A.rows(), batch);
			int cnv = conv_select(eval, ui, 0, tmpeval, tmpevec);
			if (cnv > prev)
				break;
			tmpeval.conservativeResize(batch - cnv + prev, 1);
			tmpevec.conservativeResize(A.rows(), batch - cnv + prev);
			if (i == restart)
				break;

			Map<MatrixXd> X(&V(0, Vj.cols()), A.rows(), ri.cols());
			Map<MatrixXd> U(&V(0, Vj.cols()), A.rows(), 0);
			if (eigenvectors.cols() > 0)
				new (&U) Map<MatrixXd>(&eigenvectors(0, 0), A.rows(), cnv);
			X = MatrixXd::Zero(A.rows(), ri.cols());
			specialCG(A, B, ri, U, X, eval);
			/*cout << X << endl;*/
			orthogonalization(X, U, B);
			/*cout << X << endl;*/
			orthogonalization(X, Vj, B);
			/*cout << X << endl;*/
			int dep = orthogonalization(X, B);
			/*cout << X << endl;*/
			new (&X) Map<MatrixXd>(&V(0, Vj.cols()), A.rows(), ri.cols() - dep);
			/*cout << X << endl;*/
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
void JD::specialCG(SparseMatrix<double>& A, SparseMatrix<double>& B, MatrixXd& b, Map<MatrixXd>& U, Map<MatrixXd>& X, MatrixXd& lam) {
	MatrixXd UT = U.transpose();
	for (int i = 0; i < b.cols(); ++i) {
		//取预处理矩阵为对角阵
		SparseMatrix<double> K1(A.rows(), A.cols());
		K1.reserve(A.rows());
		for (int j = 0; j < A.rows(); ++j)
			K1.insert(j, j) = 1 / (A.coeff(j, j) - lam(i, 0) * B.coeff(j, j));
		SparseMatrix<double> tmpA = A - lam(i, 0) * B;

		MatrixXd Y = K1 * U;
		MatrixXd tmpN = UT * Y;
		//TODO 不确定这里对不对
		tmpN.reverseInPlace();
		MatrixXd Ubk = tmpN * UT;

		Map<MatrixXd> zi(&X(0, i), A.rows(), 1);
		MatrixXd fi = MatrixXd::Zero(U.cols(), 1);

		MatrixXd r = MatrixXd::Zero(A.rows() + U.cols(), 1);
		Map<MatrixXd> rt(&r(0, 0), A.rows(), 1);
		rt = -b.col(i);
		rt -= K1 * tmpA * zi;
		rt -= U * fi;
		Map<MatrixXd> rb(&r(0, 0), U.cols(), 1);
		if (U.cols() > 0)
			new (&rb)Map<MatrixXd>(&r(A.rows(), 0), U.cols(), 1);
		rb -= UT * zi;

		MatrixXd z = MatrixXd::Zero(r.rows(), 1);
		Map<MatrixXd> zt(&z(0, 0), A.rows(), 1);
		Map<MatrixXd> zb(&z(0, 0), U.cols(), 1);
		if (U.cols() > 0)
			new (&zb)Map<MatrixXd>(&z(A.rows(), 0), U.cols(), 1);
		z = r;
		//TODO 验证一下原地乘结果是否正确
		zt = K1 * zt;
		zb = tmpN * (UT * zt + zb);
		zt -= Y * zb;

		MatrixXd p = MatrixXd::Zero(r.rows(), 1);
		Map<MatrixXd> pt(&p(0, 0), A.rows(), 1);
		Map<MatrixXd> pb(&p(0, 0), U.cols(), 1);
		if (U.cols() > 0)
			new (&pb)Map<MatrixXd>(&p(A.rows(), 0), U.cols(), 1);
		p = z;
		double alpha = 0;
		double beta = 0;

		MatrixXd rz = r.transpose() * z;
		/*if (i == 4)
			cout << "A：" << endl << tmpA << endl << "---------------" << endl;
		if (i == 4)
			cout << "K1：" << endl << K1 << endl << "---------------" << endl;*/

		for (int j = 0; j < cgstep; ++j) {
			MatrixXd Ap(r.rows(), 1);
			Map<MatrixXd> Apt(&Ap(0, 0), A.rows(), 1);
			Map<MatrixXd> Apb(&Ap(0, 0), U.cols(), 1);
			if (U.cols() > 0)
				new (&Apb)Map<MatrixXd>(&Ap(A.rows(), 0), U.cols(), 1);
			
			/*if (i == 4)
				cout << "p：" << endl << p << endl << "---------------" << endl;
			if (i == 4)
				cout << "z：" << endl << z << endl << "---------------" << endl;*/
			Apt = tmpA * pt + U * pb;
			Apb = UT * pt;
			/*if (i == 4)
				cout << "近似解：" << endl << Ap << endl << "---------------" << endl;*/
			alpha = rz(0, 0) / (Ap.transpose() * p)(0, 0);
			/*if (i == 4)
				cout << "alpha：" << endl << alpha << endl << "---------------" << endl;*/

			zi += alpha * pt;
			/*if (i == 4)
				cout << "近似解：" << endl << zi.block(0, 0, 10, 1) << endl << "---------------" << endl;*/
			fi += alpha * pb;
			r -= alpha * Ap;
			/*if (i == 4)
				cout << "残差：" << endl << r << endl << "---------------" << endl;*/
			/*if (r.norm() < LinearEigenSolver::EIGTOL)
				break;*/
			z = r;
			//TODO 验证一下原地乘结果是否正确
			zt = K1 * zt;
			zb = tmpN * (UT * zt + zb);
			zt -= Y * zb;
			double tmp = rz(0, 0);
			rz = r.transpose() * z;
			beta = rz(0, 0) / tmp;
			/*if (i == 4)
				cout << "beta：" << endl << beta << endl << "---------------" << endl;*/
			p *= beta;
			p += z;
		}
	}

}