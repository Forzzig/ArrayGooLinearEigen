#ifndef __TIMECONTROL_H__
#define __TIMECONTROL_H__

//��ʱ����
#define time_tol 18000
#define total_time_tol 18000

#include<chrono>
#include<ctime>
extern time_t current;

bool timeCheck(time_t& st, time_t& en);

bool totalTimeCheck(time_t& st, time_t& en);

#endif // !__TIMECONTROL_H__
