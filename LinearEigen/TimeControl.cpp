#include<TimeControl.h>
#include<iostream>

using namespace std;

bool timeCheck(time_t& st, time_t& en) {
	cout << endl << "��������ѽ���" << en - st << "��" << endl << endl;
	if (en - st > time_tol)
		return true;
	return false;
}

bool totalTimeCheck(time_t& st, time_t& en) {
	cout << endl << "��ǰ��������ѽ���" << en - st << "��" << endl << endl;
	if (en - st > total_time_tol)
		return true;
	return false;
}