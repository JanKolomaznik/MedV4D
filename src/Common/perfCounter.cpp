
#include <iostream>
#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/perfCounter.h"


///////////////////////////////////////////////////////////////////////////////

std::ostream& operator<<(std::ostream &os, PerfCounter &pc)
{
	return os << ( (float32) pc.total_ / (float32) CLOCKS_PER_SEC ) << " s";
}

///////////////////////////////////////////////////////////////////////////////
PerfCounter::PerfCounter()
{
	Reset();
}
///////////////////////////////////////////////////////////////////////////////
void
PerfCounter::Reset(void)
{
	total_ = time_start_ = 0; 
}
///////////////////////////////////////////////////////////////////////////////

void PerfCounter::Start(void)
{
	time_start_ = clock();
}
///////////////////////////////////////////////////////////////////////////////
void PerfCounter::Stop(void)
{
	total_ += clock() - time_start_;
	time_start_ = 0;
}
///////////////////////////////////////////////////////////////////////////////
