#include<TimeControl.h>
#include<iostream>

using namespace std;

bool timeCheck(time_t& st, time_t& en) {
	cout << endl << "本次求解已进行" << en - st << "秒" << endl << endl;
	if (en - st > time_tol)
		return true;
	return false;
}

bool totalTimeCheck(time_t& st, time_t& en) {
	cout << endl << "当前矩阵计算已进行" << en - st << "秒" << endl << endl;
	if (en - st > total_time_tol)
		return true;
	return false;
}