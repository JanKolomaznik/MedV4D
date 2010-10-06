/***********************************
	CTIMER
************************************/

#include "timer.h"

#ifdef WIN32
#include <windows.h>
#else

#include <sys/time.h>

unsigned GetTickCount() {
	struct timeval tv;
	if(gettimeofday(&tv, 0) != 0)
		return 0;

	return (tv.tv_sec * 1000) + (tv.tv_usec / 1000);
}

#endif


Timer::Timer() {
	restart();
}

Timer::~Timer() {
	
}

void Timer::restart() {
	timer = GetTickCount();
	measuredTime = 0;
}

void Timer::measure() {
	measuredTime = GetTickCount() - timer;
}

unsigned int Timer::getTime() {
	return measuredTime;
}

void Timer::getTime(int &hours, int &minutes, int &seconds, int &millis) {
	Timer::convertTime(measuredTime, hours, minutes, seconds, millis);
}

void Timer::convertTime(unsigned int totalMillis, int &hours, int &minutes, int &seconds, int &millis){ 
	millis = totalMillis;
	seconds = millis / 1000;
	millis -= seconds * 1000;
	minutes = seconds / 60;
	seconds -= minutes * 60;
	hours = minutes / 60;
	minutes -= hours * 60;
}

bool Timer::enabled() {
	return true;
/*#ifdef WIN32
	return true;
#else
	return false;
#endif*/
}

