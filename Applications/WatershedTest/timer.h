/***********************************
	CTIMER
************************************/

#ifndef _TIMER
#define _TIMER

class Timer {
	unsigned int timer;
	unsigned int measuredTime;
public:
	// restarts the timer
	void restart();
	// saves time from last restart
	void measure();
	// returns time from last measurement
	unsigned int getTime();
	// returns formatted time from last measurement
	void getTime(int &hours, int &minutes, int &seconds, int &millis);

	static bool enabled();

	static void convertTime(unsigned int totalMillis, int &hours, int &minutes, int &seconds, int &millis);
public:
	Timer();
	~Timer();
};

#endif
