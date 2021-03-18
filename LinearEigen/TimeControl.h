#ifndef __TIMECONTROL_H__
#define __TIMECONTROL_H__

//³¬Ê±ÏÞÖÆ
#define time_tol 1200
#define total_time_tol 3600

#include<chrono>
#include<ctime>
extern time_t current;

bool timeCheck(time_t& st, time_t& en);

bool totalTimeCheck(time_t& st, time_t& en);

#endif // !__TIMECONTROL_H__
