#ifndef __BJD_H__
#define __BJD_H__
#include<LinearEigenSolver.h>
#include<mgmres.hpp>

//#define DIAG_PRECOND

using namespace Eigen;
class BJD : public LinearEigenSolver {
public:
	int restart, gmres_size, gmres_restart, nRestart;
	MatrixXd V/*, W*/, HA/*, HB*/;
    MatrixXd Atmp, Btmp, AV/*, BV*/;
	vector<MatrixXd> Y, v;

#ifdef DIAG_PRECOND
    vector<vector<double>> Ainv;
#endif // DIAG_PRECOND

    int batch;
    //double tau;

    GeneralizedSelfAdjointEigenSolver<MatrixXd> ges;

	void CRSsort(int* ia, int* ja, double* a, int n);
	void genCRS(SparseMatrix<double, RowMajor, __int64>& A, int* ia, int* ja, double* a);

	void CRSsubtrac(int*& ia, int*& ja, double*& a, int& nnz, SparseMatrix<double, RowMajor, __int64>& B, double eff);

	BJD(SparseMatrix<double, RowMajor, __int64>& A, SparseMatrix<double, RowMajor, __int64>& B, int nev, int restart, int batch, int gmres_size, int gmres_restart);
	~BJD();

	////用来实现求解V'(A-tauB)'AV = lam*V'(A-tauB)'BV
	//template<typename Derived_HA, typename Derived_HB, typename Derived_val, typename Derived_vec>
	//void generalized_RR(Derived_HA& HA, Derived_HB& HB, Derived_val& eigenvalues, Derived_vec& eigenvectors) {
	//	ges.compute(HA, HB);
	//	com_of_mul += 24 * HA.cols() * HA.cols() * HA.cols();
	//	eigenvalues = ges.eigenvalues();
	//	eigenvectors = ges.eigenvectors();
	//}
	
	template<typename Derived_U, typename Derived_Y, typename Derived_v>
	void Minv_set(SparseMatrix<double, RowMajor, __int64>& A, SparseMatrix<double, RowMajor, __int64> &B, double lam, Derived_U& U, Derived_Y& Y, Derived_v& v, vector<double> &Ainv) {
		/*[Ki, U;]^(-1)
		   UT, 0;      * r = z
		   K1 = Ki^(-1)        */

        if (Y.cols() != U.cols())
            Y.resize(NoChange, U.cols());
        for (int i = 0; i < A.rows(); ++i) {
            double diag = A.coeff(i, i) - lam * B.coeff(i, i);
            if (abs(diag) > ORTH_TOL)
                Ainv[i] = 1.0 / diag;
            else
                Ainv[i] = ((diag < 0) ? -1.0 : 1.0) / ORTH_TOL;
            Y.row(i).noalias() = Ainv[i] * U.row(i);
        }
		com_of_mul += 5 * A.rows() + U.rows() * U.cols();

        if (v.cols() != Y.cols())
            v.resize(Y.cols(), Y.cols());
		v.noalias() = U.transpose() * Y;
		com_of_mul += U.cols() * U.rows() * Y.cols();

		v = v.inverse();
		com_of_mul += v.cols() * v.cols() * v.cols() * 2 / 3 + 7 * v.cols() * v.cols() - v.cols() - v.cols();
	}

    //左乘带U的预优矩阵，r和z可以相同
	template<typename Derived_U, typename Derived_Y, typename Derived_v, typename Derived_r, typename Derived_z>
	void Minv_mul(Derived_U& U, Derived_Y& Y, Derived_v& v, vector<double> &Ainv, Derived_r& r, Derived_z& z) {
		/*[Ki, U;]^(-1)
		   UT, 0;      * r = z
		   K1 = Ki^(-1)        
        ================================
         [I -Y] * [I   ] * [I    ] * [K1  ] * r = z
             I       -v     -UT I        I
        */

        for (int i = 0; i < A.rows(); ++i)
            z.row(i) = r.row(i) * Ainv[i];
        com_of_mul += A.rows() * z.cols();

        z.bottomRows(U.cols()) = U.transpose() * z.topRows(A.rows()) - r.bottomRows(U.cols());
        z.bottomRows(U.cols()).applyOnTheLeft(v);
        com_of_mul += (v.rows() * v.cols() + U.cols() * U.rows()) * z.cols();

		z.topRows(A.rows()) -= Y * z.bottomRows(U.cols());
		com_of_mul += Y.rows() * Y.cols();
	}

	//左乘A对应的增广矩阵, r和z不能相同
	template<typename Derived_U, typename Derived_r, typename Derived_z, typename Derived_tmp>
	void A_mul(SparseMatrix<double, RowMajor, __int64>& A, SparseMatrix<double, RowMajor, __int64>& B, double lam, Derived_U& U, Derived_r& r, Derived_z& z, Derived_tmp& Atmp, Derived_tmp& Btmp) {
        Atmp.noalias() = A * r.topRows(A.rows());
        Btmp.noalias() = B * r.topRows(A.rows());
        z.topRows(A.rows()).noalias() = U * r.bottomRows(U.cols());
        z.topRows(A.rows()) += Atmp;
        z.topRows(A.rows()) -= lam * Btmp;
		com_of_mul += (A.nonZeros() + B.nonZeros()) * r.cols() + U.rows() * U.cols() * r.cols() + A.rows();
		
		z.bottomRows(U.cols()).noalias() = U.transpose() * r.topRows(A.rows());
		com_of_mul += U.cols() * U.rows();
	}
	
	/*
    //左预处理GMRES，BJD特化版本
	template<typename Derived_rhs, typename Derived_ss, typename Derived_sol, typename Derived_eval>
	void L_GMRES(SparseMatrix<double, RowMajor, __int64>& A, SparseMatrix<double, RowMajor, __int64>& B, Derived_rhs& b, Derived_ss& U, Derived_sol& X, Derived_eval& lam, int m) {
		//(求解：（I-UUT）（A-λB）z = b)
		K1.resize(A.rows(), A.cols());
		K1.reserve(A.rows());
		for (int j = 0; j < A.rows(); ++j)
			K1.insert(j, j) = 1;

		MatrixXd x(A.rows() + U.cols(), 1);
		MatrixXd r(x.rows(), 1);
		MatrixXd tpw(x.rows(), 1);
		MatrixXd V(x.rows(), m);
		MatrixXd H(m + 1, m);
		MatrixXd w(x.rows(), 1);
		MatrixXd y;

		for (int i = 0; i < b.cols(); ++i) {

			//注意K逆的生成
			if (i == 0) {
				tmpA = A;
				tmpA -= lam(0, 0) * B;
			}
			else {
				tmpA -= (lam(i, 0) - lam(i - 1, 0)) * B;
			}
			com_of_mul += B.nonZeros();

			//TODO 避免0对角元,也可用其他方法
			//先取预处理矩阵为对角阵
			for (int j = 0; j < A.rows(); ++j)
				if (abs(tmpA.coeff(j, j)) < LinearEigenSolver::ORTH_TOL)
					K1.coeffRef(j, j) = 1 / LinearEigenSolver::ORTH_TOL;
				else
					K1.coeffRef(j, j) = 1 / tmpA.coeff(j, j);
			com_of_mul += A.rows() * 5;

			Minv_set(U);
			x.topRows(tmpA.rows()) = X.col(i);
			x.bottomRows(U.cols()) = MatrixXd::Zero(U.cols(), 1);

			//由于CG和GMRES不能等价，因此用CG中的cgstep近似控制一下计算量
			//这样cgstep仍然代表大矩阵乘法进行的次数，m为GMRES扩展子空间的大小，maxit为GMRES重启次数
			int maxit = cgstep / m;
			for (int it = 0; it < maxit; ++it) {

				A_mul(U, x, r);
				r.topRows(tmpA.rows()) -= b.col(i);
				tpw = -r;
				Minv_mul(U, tpw, r);

				double beta = r.norm();
				V.col(0) = r / beta;
				com_of_mul += 2 * A.rows() + 2 * U.cols();

				//错误就在这里，细心的你能看到吗？
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
					com_of_mul += (j + 1) * V.rows() * 2 + V.rows();

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
				com_of_mul += H.rows() * 30 + 2 * H.rows() * H.rows();

				for (int j = H.rows() - 2; j >= 0; --j) {
					y(j, 0) /= H(j, j);
					for (int k = j - 1; k >= 0; --k) {
						y(k, 0) -= H(k, j) * y(j, 0);
					}
				}
				com_of_mul += (H.rows() + 1) * H.rows() / 2 + 5 * H.rows();

				x += V * y.topRows(V.cols());
				com_of_mul += V.rows() * V.cols();

				if (abs(y(y.rows() - 1, 0)) < LinearEigenSolver::ORTH_TOL)
					break;
			}
			X.col(i) = x.topRows(A.rows());
		}
	}
    */

    template<typename Derived_rhs, typename Derived_ss, typename Derived_sol>
    void PMGMRES(SparseMatrix<double, RowMajor, __int64>& A, SparseMatrix<double, RowMajor, __int64>& B, Derived_rhs& b, Derived_ss& U, Derived_sol& X, double lam, int tmpindex) {

        double av;
        double* c;
        double delta = 1.0e-03;
        double* g;
        double* h;
        double htmp;
        int i;
        int itr;
        int itr_used;
        int j;
        int k;
        int k_copy;
        double mu;
        double* r;
        double rho;
        double rho_tol;
        double* s;
        double* v;
        int verbose = 0;
        double* y;
        //double* l;
        //int* ua;

        int n = A.rows() + U.cols();
        c = new double[gmres_size + 1];
        g = new double[gmres_size + 1];
        h = new double[(gmres_size + 1) * gmres_size];
        r = new double[n];
        s = new double[gmres_size + 1];
        v = new double[(gmres_size + 1) * n];
        y = new double[gmres_size + 1];
        //l = new double[ia[A.rows()] + 1];
        //ua = new int[A.rows()];

        Map<MatrixXd> R(r, n, 1);
        Map<MatrixXd> V(v, n, gmres_size + 1);
        Map<MatrixXd> Y(y, gmres_size + 1, 1);

        VectorXd x(A.rows() + U.cols());

        itr_used = 0;

        x.topRows(A.rows()) = X;
        x.bottomRows(U.cols()) = MatrixXd::Zero(U.cols(), 1);

#ifdef DIAG_PRECOND
        Minv_set(A, B, lam, U, this->Y[tmpindex], this->v[tmpindex], Ainv[tmpindex]);
#endif // DIAG_PRECOND

        if (verbose)
        {
            cout << "\n";
            cout << "PMGMRES_ILU_CR\n";
            cout << "  Number of unknowns = " << n << "\n";
        }

        for (itr = 0; itr < gmres_restart; itr++)
        {
            A_mul(A, B, lam, U, x, R, this->Atmp.col(tmpindex), this->Btmp.col(tmpindex));
            R *= -1;
            R.topRows(A.rows()) += b;
            com_of_mul += A.rows();

#ifdef DIAG_PRECOND
            Minv_mul(U, this->Y[tmpindex], this->v[tmpindex], Ainv[tmpindex], R, R);
#endif // DIAG_PRECOND

            //rho = sqrt(r8vec_dot(n, r, r));
            rho = R.norm();

            if (verbose)
            {
                cout << "  ITR = " << itr << "  Residual = " << rho << "\n";
            }

            if (itr == 0)
            {
                rho_tol = rho * LinearEigenSolver::ORTH_TOL;
            }

            V.col(0) = R / rho;

            g[0] = rho;
            for (i = 1; i < gmres_size + 1; i++)
            {
                g[i] = 0.0;
            }

            for (i = 0; i < gmres_size + 1; i++)
            {
                for (j = 0; j < gmres_size; j++)
                {
                    h[i * (gmres_size)+j] = 0.0;
                }
            }

            for (k = 0; k < gmres_size; k++)
            {
                k_copy = k;

                A_mul(A, B, lam, U, V.col(k), V.col(k + 1), this->Atmp.col(tmpindex), this->Btmp.col(tmpindex));
#ifdef DIAG_PRECOND
                Minv_mul(U, this->Y[tmpindex], this->v[tmpindex], Ainv[tmpindex], V.col(k + 1), V.col(k + 1));
#endif // DIAG_PRECOND

                //A_mul(U, V.col(k), V.col(k + 1));
                //Minv_mul(U, V.col(k + 1), V.col(k + 1));

                //av = sqrt(r8vec_dot(n, v + (k + 1) * n, v + (k + 1) * n));
                av = V.col(k + 1).norm();
                com_of_mul += n;

                for (j = 0; j <= k; j++)
                {
                    //h[j * gmres_size + k] = r8vec_dot(n, v + (k + 1) * n, v + j * n);
                    h[j * gmres_size + k] = V.col(k + 1).dot(V.col(j));
                    V.col(k + 1) -= h[j * gmres_size + k] * V.col(j);
                }
                com_of_mul += n * (k + 1) * 2;

                //h[(k + 1) * gmres_size + k] = sqrt(r8vec_dot(n, v + (k + 1) * n, v + (k + 1) * n));
                h[(k + 1) * gmres_size + k] = V.col(k + 1).norm();
                com_of_mul += n;

                if ((av + delta * h[(k + 1) * gmres_size + k]) == av)
                {
                    for (j = 0; j < k + 1; j++)
                    {
                        //htmp = r8vec_dot(n, v + (k + 1) * n, v + j * n);
                        htmp = V.col(k + 1).dot(V.col(j));
                        com_of_mul += n;

                        h[j * gmres_size + k] = h[j * gmres_size + k] + htmp;
                        V.col(k + 1) = V.col(k + 1) - htmp * V.col(j);
                        com_of_mul += n;
                    }
                    //h[(k + 1) * gmres_size + k] = sqrt(r8vec_dot(n, v + (k + 1) * n, v + (k + 1) * n));
                    h[(k + 1) * gmres_size + k] = V.col(k + 1).norm();
                    com_of_mul += n;
                }

                if (h[(k + 1) * gmres_size + k] != 0.0)
                {
                    V.col(k + 1) = V.col(k + 1) / h[(k + 1) * gmres_size + k];
                    com_of_mul += 5 * n;
                }

                if (0 < k)
                {
                    for (i = 0; i < k + 2; i++)
                    {
                        y[i] = h[i * gmres_size + k];
                    }
                    for (j = 0; j < k; j++)
                    {
                        mult_givens(c[j], s[j], j, y);
                    }
                    for (i = 0; i < k + 2; i++)
                    {
                        h[i * gmres_size + k] = y[i];
                    }
                }
                mu = sqrt(h[k * gmres_size + k] * h[k * gmres_size + k] + h[(k + 1) * gmres_size + k] * h[(k + 1) * gmres_size + k]);
                c[k] = h[k * gmres_size + k] / mu;
                s[k] = -h[(k + 1) * gmres_size + k] / mu;
                h[k * gmres_size + k] = c[k] * h[k * gmres_size + k] - s[k] * h[(k + 1) * gmres_size + k];
                h[(k + 1) * gmres_size + k] = 0.0;
                mult_givens(c[k], s[k], k, g);

                rho = fabs(g[k + 1]);

                itr_used = itr_used + 1;

                if (verbose)
                {
                    cout << "  K   = " << k << "  Residual = " << rho << "\n";
                }

                if (rho <= rho_tol && rho <= sqrt(LinearEigenSolver::ORTH_TOL))
                {
                    break;
                }
            }

            k = k_copy;

            y[k] = g[k] / h[k * gmres_size + k];
            for (i = k - 1; 0 <= i; i--)
            {
                y[i] = g[i];
                for (j = i + 1; j < k + 1; j++)
                {
                    y[i] = y[i] - h[i * gmres_size + j] * y[j];
                }
                y[i] = y[i] / h[i * gmres_size + i];
            }
            com_of_mul += k * (k + 1) / 2;

            x += V.leftCols(k + 1) * Y.topRows(k + 1);
            com_of_mul += V.rows() * (k + 1);

            if (rho <= rho_tol && rho <= sqrt(LinearEigenSolver::ORTH_TOL))
            {
                break;
            }
        }
        X = x.topRows(A.rows());

        if (verbose)
        {
            cout << "\n";;
            cout << "PMGMRES_ILU_CR:\n";
            cout << "  Iterations = " << itr_used << "\n";
            cout << "  Final residual = " << rho << "\n";
        }

        //
        //  Free memory.
        //
        delete[] c;
        delete[] g;
        delete[] h;
        delete[] r;
        delete[] s;
        delete[] v;
        delete[] y;
        //delete[] l;
        //delete[] ua;
    }
	void compute();
};

#endif
