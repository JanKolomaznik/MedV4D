#ifndef TIME_MEASUREMENT_H
#define TIME_MEASUREMENT_H

#include <ctime>
#include <boost/date_time/posix_time/posix_time.hpp>

namespace M4D
{
namespace Common
{

class Clock
{
public:
	Clock(): mTime( boost::posix_time::microsec_clock::local_time() )
	{ }

	double
	SecondsPassed()const //deprecated name
	{
		return secondsPassed();
	}

	double
	secondsPassed()const
	{
		boost::posix_time::time_duration td = boost::posix_time::microsec_clock::local_time() - mTime;
		return 1.0E-9 * double(td.total_nanoseconds());//return double(::clock() - mTime)/CLOCKS_PER_SEC;
	}

	void
	Reset()
	{
		mTime = boost::posix_time::microsec_clock::local_time();
	}
private:
	boost::posix_time::ptime mTime;
};

}/*namespace Common*/
}/*namespace M4D*/

#endif /*TIME_MEASUREMENT_H*/
