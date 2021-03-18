#include<TimeControl.h>

bool timeCheck(time_t& st, time_t& en) {
	if (en - st > time_tol)
		return true;
	return false;
}

bool totalTimeCheck(time_t& st, time_t& en) {
	if (en - st > total_time_tol)
		return true;
	return false;
}