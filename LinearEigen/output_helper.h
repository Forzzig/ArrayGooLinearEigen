#ifndef __OUTPUT_HELPER_H__
#define __OUTPUT_HELPER_H__

#include<fstream>
#include<string>
#include<direct.h>
#include<iostream>
#include<TimeControl.h>
#include<ctime>
#include<Eigen/Sparse>
#include<iomanip>

using namespace std;
using namespace Eigen;

int fstream_prepare(fstream& txt, fstream& csv, SparseMatrix<double>& A, string& matrix_name, string& method, string& suff) {
	string folderPath = "./result/" + matrix_name;
	_mkdir(folderPath.c_str());

	//TODO С�������ʽ����

	cout << "��" << matrix_name << "ʹ��" + method + "........................................." << endl;
	txt.open(folderPath + "/" + matrix_name + "-" + method + suff + ".txt", ios::out | ios::app);

	current = time(&current);
	char buff[26];
	ctime_s(buff, sizeof(buff), &current);
	txt << endl << buff;

	txt << "���������" << A.rows() << "��" << "����Ԫ����" << A.nonZeros() << endl << endl;
	txt << matrix_name + "��" + method + "��ʼ���........................................." << endl;

	csv.open(folderPath + "/" + matrix_name + "-" + method + "-statistics" + suff + ".csv", ios::out | ios::app);
	csv << endl << buff;
	return 0;
}

#endif
