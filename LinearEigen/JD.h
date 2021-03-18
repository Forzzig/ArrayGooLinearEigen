#ifndef __JD_H__
#define __JD_H__
#include<LinearEigenSolver.h>

using namespace Eigen;
class JD : public LinearEigenSolver {
public:
	int restart, gmres_size;
	MatrixXd W, H, U;
	Map<MatrixXd> V;
	MatrixXd Y, v;
	SparseMatrix<double> K1, tmpA;
	int cgstep;
	int batch;
	JD(SparseMatrix<double>& A, SparseMatrix<double>& B, int nev, int cgstep, int restart, int batch, int gmres_size);
	
	template<typename Derived_rhs, typename Derived_ss, typename Derived_sol, typename Derived_eval>
	void specialCG(SparseMatrix<double>& A, SparseMatrix<double>& B, Derived_rhs& b, Derived_ss& U, Derived_sol& X, Derived_eval& lam) {
		//(求解：（I-UUT）（A-λB）z = b)

		MatrixXd UT = U.transpose();
		//取预处理矩阵为对角阵
		SparseMatrix<double> K1(A.rows(), A.cols());
		K1.reserve(A.rows());
		for (int j = 0; j < A.rows(); ++j)
			K1.insert(j, j) = 1 / A.coeff(j, j);
		SparseMatrix<double> tmpA = A;

		for (int i = 0; i < b.cols(); ++i) {
			
			if (i == 0) {
				tmpA -= lam(0, 0) * B;
			}
			else {
				tmpA -= (lam(i, 0) - lam(i - 1, 0)) * B;
			}
			for (int j = 0; j < A.rows(); ++j)
				K1.coeffRef(j, j) = 1 / tmpA.coeff(j, j);

			MatrixXd Y = K1 * U;
			MatrixXd tmpN = UT * Y;
			//cout << tmpN << endl;
			tmpN = tmpN.inverse();
			//cout << tmpN << endl;
			
			MatrixXd pku(A.rows() + U.cols(), A.cols() + U.cols());
			pku.block(0, 0, A.rows(), A.cols()) = A;
			pku.block(0, A.cols(), U.rows(), U.cols()) = U;
			pku.block(A.rows(), 0, U.cols(), U.rows()) = U.transpose();
			pku.block(A.rows(), A.cols(), U.cols(), U.cols()) = MatrixXd::Zero(U.cols(), U.cols());

			//cout << "-----------------------" << endl;
			//cout << pku << endl;
			//system("pause");
			Map<MatrixXd> zi(&X(0, i), A.rows(), 1);
			
			//TODO fi可能不需要
			MatrixXd fi = MatrixXd::Zero(U.cols(), 1);

			//cout << "-----------------------" << endl;
			//cout << zi << endl;
			//cout << fi << endl;

			MatrixXd pkutmp(A.rows() + U.cols(), 1);
			pkutmp.block(0, 0, A.rows(), 1) = zi;
			pkutmp.block(A.rows(), 0, U.cols(), 1) = fi;
			MatrixXd pkurhs(A.rows() + U.cols(), 1);
			pkurhs.block(0, 0, A.rows(), 1) = b.col(i);
			pkurhs.block(A.rows(), 0, U.cols(), 1) = MatrixXd::Zero(U.cols(), 1);
			
			//cout << "-----------------------" << endl;
			//cout << pkutmp << endl;
			//system("pause");

			MatrixXd pkuM(A.rows() + U.cols(), A.cols() + U.cols());
			pkuM.block(0, 0, A.rows(), A.cols()) = MatrixXd::Identity(A.rows(), A.rows());
			pkuM.block(0, A.cols(), U.rows(), U.cols()) = -Y;
			pkuM.block(A.rows(), 0, U.cols(), U.rows()) = MatrixXd::Zero(U.cols(), A.rows());
			pkuM.block(A.rows(), A.cols(), U.cols(), U.cols()) = MatrixXd::Identity(U.cols(), U.cols());
			MatrixXd uttmp(A.rows() + U.cols(), A.cols() + U.cols());
			uttmp.block(0, 0, A.rows(), A.cols()) = MatrixXd::Identity(A.rows(), A.rows());
			uttmp.block(0, A.cols(), U.rows(), U.cols()) = MatrixXd::Zero(A.rows(), U.cols());
			uttmp.block(A.rows(), 0, U.cols(), U.rows()) = tmpN * UT;
			uttmp.block(A.rows(), A.cols(), U.cols(), U.cols()) = tmpN;
			pkuM *= uttmp;
			uttmp.block(0, 0, A.rows(), A.cols()) = K1;
			uttmp.block(0, A.cols(), U.rows(), U.cols()) = MatrixXd::Zero(A.rows(), U.cols());
			uttmp.block(A.rows(), 0, U.cols(), U.rows()) = MatrixXd::Zero(U.cols(), A.rows());
			uttmp.block(A.rows(), A.cols(), U.cols(), U.cols()) = MatrixXd::Identity(U.cols(), U.cols());
			pkuM *= uttmp;
			
			MatrixXd r = MatrixXd::Zero(A.rows() + U.cols(), 1);
			Map<MatrixXd> rt(&r(0, 0), A.rows(), 1);
			rt = b.col(i);
			rt -= tmpA * zi;
			Map<MatrixXd> rb(&r(0, 0), U.cols(), 1);
			if (U.cols() > 0)
				new (&rb)Map<MatrixXd>(&r(A.rows(), 0), U.cols(), 1);
			rb -= UT * zi;

			//cout << "-----------------------" << endl;
			//cout << r << endl;
			//cout << r.norm() << endl;
			//cout << "残差：" << (r - (pkurhs - pku * pkutmp)).norm() << endl;
			//system("pause");

			MatrixXd z = r;
			Map<MatrixXd> zt(&z(0, 0), A.rows(), 1);
			Map<MatrixXd> zb(&z(0, 0), U.cols(), 1);
			if (U.cols() > 0)
				new (&zb)Map<MatrixXd>(&z(A.rows(), 0), U.cols(), 1);
			zt.applyOnTheLeft(K1);

			//cout << "-----------------------" << endl;
			//cout << z << endl;
			//system("pause");

			zb += UT * zt;
			zb.applyOnTheLeft(tmpN);

			//cout << "-----------------------" << endl;
			//cout << z << endl;
			//system("pause");

			zt -= Y * zb;

			//cout << "-----------------------" << endl;
			//cout << z << endl;
			//cout << "预优残差：" << (z - pkuM * r).norm() << endl;
			//system("pause");

			MatrixXd p = z;
			Map<MatrixXd> pt(&p(0, 0), A.rows(), 1);
			Map<MatrixXd> pb(&p(0, 0), U.cols(), 1);
			if (U.cols() > 0)
				new (&pb)Map<MatrixXd>(&p(A.rows(), 0), U.cols(), 1);
			double alpha = 0;
			double beta = 0;

			//cout << "-----------------------" << endl;
			//cout << p << endl;
			//system("pause");

			double rz = (r.transpose() * z)(0, 0);
			double bflag = b.norm();
			
			for (int j = 0; j < cgstep; ++j) {
				
				MatrixXd Ap(r.rows(), 1);
				Map<MatrixXd> Apt(&Ap(0, 0), A.rows(), 1);
				Map<MatrixXd> Apb(&Ap(0, 0), U.cols(), 1);
				if (U.cols() > 0)
					new (&Apb)Map<MatrixXd>(&Ap(A.rows(), 0), U.cols(), 1);

				Apt = tmpA * pt + U * pb;
				Apb = UT * pt;

				MatrixXd tmprz = (pkurhs - pku * pkutmp).transpose() * pkuM * (pkurhs - pku * pkutmp);

				alpha = rz / (p.transpose() * Ap)(0, 0);

				/*cout << rz << endl;
				cout << (p.transpose() * Ap)(0, 0) << endl;
				cout << "alpha："<<alpha << endl;
				cout << "it should be：" << tmprz(0, 0) / (p.transpose() * Ap)(0, 0) << endl;*/
				//system("pause");

				zi += alpha * pt;
				//fi += alpha * pb;

				/*cout << "-----------------------" << endl;
				cout << zi << endl;
				cout << fi << endl;
				system("pause");*/

				r -= alpha * Ap;

				//cout << "-----------------------" << endl;
				////cout << r << endl;
				//pkutmp.block(0, 0, A.rows(), 1) = zi;
				//pkutmp.block(A.rows(), 0, U.cols(), 1) = fi;
				//cout << "残差：" << r.norm() << endl;
				//cout << "残差误差：" << (rt - (b - A * zi)).norm() << endl;
				//system("pause");
				if (r.norm() / bflag < LinearEigenSolver::EIGTOL)
					break;

				z = r;

				zt.applyOnTheLeft(K1);
				zb += UT * zt;
				zb.applyOnTheLeft(tmpN);
				zt -= Y * zb;

				/*cout << "-----------------------" << endl;
				cout << z << endl;
				cout << "预优残差：" << z.norm() << endl;
				cout << "预优残差误差：" << (zt - pkuM * (pkurhs - pku * pkutmp)).norm() << endl;
				system("pause");*/
				
				double tmp = rz;
				rz = (r.transpose() * z)(0, 0);
				beta = rz / tmp;


				//cout << rz << endl;
				//cout << tmp << endl;
				//cout << beta << endl;
				//system("pause");
					
				p *= beta;
				p += z;

				/*cout << "-----------------------" << endl;
				cout << p << endl;*/
				//system("pause");
			}
		}

	}
	
	template<typename Derived_U>
	void Minv_set(Derived_U& U) {
		/*[Ki, U;]^(-1)
		   UT, 0;      * r = z
		   K1 = Ki^(-1)        */
		Y = K1 * U;
		v = U.transpose() * Y;
		v = v.inverse();
	}
	
	template<typename Derived_U, typename Derived_r, typename Derived_z>
	void Minv_mul(Derived_U &U, Derived_r& r, Derived_z& z) {
		/*[Ki, U;]^(-1)
		   UT, 0;      * r = z
		   K1 = Ki^(-1)        */
		z.block(0, 0, K1.rows(), 1) = K1 * r.block(0, 0, K1.rows(), 1);
		z.block(K1.rows(), 0, U.cols(), 1) = v * (r.block(K1.rows(), 0, U.cols(), 1) + U.transpose() * z.block(0, 0, K1.rows(), 1));
		z.block(0, 0, K1.rows(), 1) -= Y * z.block(K1.rows(), 0, U.cols(), 1);
	}
	template<typename Derived_U, typename Derived_r, typename Derived_z>
	void A_mul(Derived_U &U, Derived_r& r, Derived_z& z) {
		z.block(0, 0, tmpA.rows(), 1) = tmpA * r.block(0, 0, tmpA.rows(), 1) + U * r.block(tmpA.rows(), 0, U.cols(), 1);
		z.block(tmpA.rows(), 0, U.cols(), 1) = U.transpose() * r.block(0, 0, tmpA.rows(), 1);
	}
	template<typename Derived_rhs, typename Derived_ss, typename Derived_sol, typename Derived_eval>
	void L_GMRES(SparseMatrix<double>& A, SparseMatrix<double>& B, Derived_rhs& b, Derived_ss& U, Derived_sol& X, Derived_eval& lam, int m) {
		//(求解：（I-UUT）（A-λB）z = b)
		K1.resize(A.rows(), A.cols());
		K1.reserve(A.rows());
		for (int j = 0; j < A.rows(); ++j)
			K1.insert(j, j) = 1;
		tmpA = A;

		//注意K逆的生成
		/*if (i == 0) {
				tmpA -= lam(0, 0) * B;
			}
			else {
				tmpA -= (lam(i, 0) - lam(i - 1, 0)) * B;
			}*/
		//取预处理矩阵为对角阵
		for (int j = 0; j < A.rows(); ++j)
			K1.coeffRef(j, j) = 1 / tmpA.coeff(j, j);
		Minv_set(U);

		MatrixXd x(tmpA.rows() + U.cols(), 1);
		MatrixXd r(x.rows(), 1);
		MatrixXd tpw(x.rows(), 1);
		MatrixXd V(x.rows(), m);
		MatrixXd H(m + 1, m);
		MatrixXd y;

		for (int i = 0; i < b.cols(); ++i) {
			
			x.block(0, 0, tmpA.rows(), 1) = X.col(i);
			x.block(tmpA.rows(), 0, U.cols(), 1) = MatrixXd::Zero(U.cols(), 1);

			int maxit = cgstep / m;
			for (int it = 0; it < maxit; ++it) {
				
				A_mul(U, x, r);
				r.block(0, 0, tmpA.rows(), 1) -= b.col(i);
				tpw = -r;
				Minv_mul(U, tpw, r);

				double beta = r.norm();
				V.col(0) = r / beta;
				MatrixXd w(x.rows(), 1);

				for (int j = 0; j < m; ++j) {
					A_mul(U, V.col(j), tpw);
					Minv_mul(U, tpw, w);
					for (int k = 0; k <= j; ++k) {
						H(k, j) = (w.transpose() * V.col(k))(0, 0);
						w -= H(k, j) * V.col(k);
					}
					H(j + 1, j) = w.norm();
					if (j < m - 1)
						V.col(j + 1) = w / H(j + 1, j);
					if (H(j + 1, j) < LinearEigenSolver::ORTH_TOL) {
						H.conservativeResize(j + 2, j + 1);
						V.conservativeResize(x.rows(), j + 1);
						break;
					}
				}

				//求解y
				y = MatrixXd::Zero(H.rows(), 1);
				y(0, 0) = beta;
				for (int j = 0; j < H.rows() - 1; ++j) {
					double tmp = sqrt(H(j, j) * H(j, j) + H(j + 1, j) * H(j + 1, j));
					double s = H(j + 1, j) / tmp;
					double c = H(j, j) / tmp;
					Matrix2d omega;
					omega << c, s,
						-s, c;
					H.block(j, j, 2, H.cols() - j).applyOnTheLeft(omega);
					y(j + 1, 0) = -s * y(j, 0);
					y(j, 0) *= c;
				}
				for (int j = H.rows() - 2; j >= 0; --j) {
					y(j, 0) /= H(j, j);
					for (int k = j - 1 ; k >= 0; --k) {
						y(k, 0) -= H(k, j) * y(j, 0);
					}
				}
				x += V * y.block(0, 0, V.cols(), 1);
				if (y(y.rows() - 1, 0) < LinearEigenSolver::ORTH_TOL)
					break;
			}
			X.col(i) = x.block(0, 0, A.rows(), 1);
		}
	}
	template<typename Derived_rhs, typename Derived_ss, typename Derived_sol, typename Derived_eval>
	void specialCG_Bad(SparseMatrix<double>& A, SparseMatrix<double>& B, Derived_rhs& b, Derived_ss& U, Derived_sol& X, Derived_eval& lam) {
		//(求解：（I-UUT）（A-λB）z = b)

		//开空间
		K1.resize(A.rows(), A.cols());
		K1.reserve(A.rows());
		for (int j = 0; j < A.rows(); ++j)
			K1.insert(j, j) = 1;
		tmpA = A;

		for (int i = 0; i < b.cols(); ++i) {
			
			if (i == 0) {
				tmpA -= lam(0, 0) * B;
			}
			else {
				tmpA -= (lam(i, 0) - lam(i - 1, 0)) * B;
			}
			//取预处理矩阵为对角阵
			for (int j = 0; j < A.rows(); ++j)
				K1.coeffRef(j, j) = 1 / tmpA.coeff(j, j);
			Minv_set(U);

			MatrixXd x(tmpA.rows() + U.cols(), 1);
			x.block(0, 0, tmpA.rows(), 1) = X.col(i);
			x.block(tmpA.rows(), 0, U.cols(), 1) = MatrixXd::Zero(U.cols(), 1);

			MatrixXd r(tmpA.rows() + U.cols(), 1);
			A_mul(U, x, r);
			r.block(0, 0, tmpA.rows(), 1) -= b.col(i);
			r = -r;


			MatrixXd z(tmpA.rows() + U.cols(), 1);
			Minv_mul(U, r, z);

			MatrixXd p = z;

			double alpha = 0;
			double beta = 0;

			/*ofstream fout("JD_out.txt");
			fout << scientific << setprecision(32);*/
			double rz = (r.transpose() * z)(0, 0);
			for (int j = 0; j < cgstep; ++j) {

				/*fout << "迭代步: " << j <<  endl;
				fout << "x:--------------------------------------------------------------------------------"
					<< endl << x << endl << "r:--------------------------------------------------------------------------------"
					<< endl << r << endl << "z:--------------------------------------------------------------------------------"
					<< endl << z << endl << "p:--------------------------------------------------------------------------------"
					<< endl << p << endl;*/
				MatrixXd Ap(tmpA.rows() + U.cols(), 1);
				A_mul(U, p, Ap);

				//cout << "残差：" << r.norm() << endl;

				alpha = rz / (p.transpose() * Ap)(0, 0);
				//cout << "alpha: " << alpha << endl;
					
				x += alpha * p;
				r -= alpha * Ap;
				//cout << "下降后残差：" << r.norm() << endl;
					
				Minv_mul(U, r, z);

				double tmp = rz;
				rz = (r.transpose() * z)(0, 0);
				beta = rz / tmp;

				p = z + beta * p;
				//cout << "p的模" << p.norm() << endl;
				//system("pause");
			}

			X.col(i) = x.block(0, 0, A.rows(), 1);
		}
		//system("pause");
	}
	void compute();
};
#endif
