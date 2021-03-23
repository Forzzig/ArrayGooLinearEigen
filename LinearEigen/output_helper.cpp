#include<output_helper.h>

int fstream_prepare(ofstream& txt, ofstream& csv, SparseMatrix<double, RowMajor>& A, string matrix_name, string method, string suff)
{
	string folderPath = "./result/";
	for (int i = 0; i < matrix_name.length(); ++i) {
		if (matrix_name[i] == '/') {
			folderPath += matrix_name.substr(0, i);
			_mkdir(folderPath.c_str());
			folderPath += '/';
			matrix_name = matrix_name.substr(i + 1, matrix_name.length() - i - 1);
			i = -1;
		}
	}
	folderPath += matrix_name;
	_mkdir(folderPath.c_str());

	//TODO 小数输出格式控制

	cout << "对" << matrix_name << "使用" + method + "........................................." << endl;
	txt.open(folderPath + "/" + matrix_name + "-" + method + suff + ".txt", ios::out | ios::app);

	current = time(&current);
	char buff[26];
	ctime_s(buff, sizeof(buff), &current);
	txt << endl << buff;

	txt << "矩阵阶数：" << A.rows() << "，" << "非零元数：" << A.nonZeros() << endl << endl;
	txt << matrix_name + "，" + method + "开始求解........................................." << endl;

	csv.open(folderPath + "/" + matrix_name + "-" + method + "-statistics" + suff + ".csv", ios::out | ios::app);
	csv << endl << buff;
	return 0;
}
