#ifndef PERFCOUNTER_H_
#define PERFCOUNTER_H_

#include <stdio.h>
#include <time.h>


class PerfCounter
{
public:
	PerfCounter();
	void Reset(void);
	
	void Start(void);
	void Stop(void);
private:
	clock_t time_start_;
	clock_t total_;

	friend std::ostream& operator<<(std::ostream &os, PerfCounter &pc);
};

#endif /*PERFCOUNTER_H_*/
