#ifndef TIME_MEASUREMENT_H
#define TIME_MEASUREMENT_H

#include <ctime>

namespace M4D
{
namespace Common
{

class Clock
{
public:
	Clock(): mTime( ::clock() )
	{}

	float32
	SecondsPassed()const
	{
		return ((float32)::clock() - mTime)/CLOCKS_PER_SEC;
	}

	void
	Reset()
	{
		mTime = ::clock();
	}
private:
	clock_t mTime;
};

}/*namespace Common*/
}/*namespace M4D*/

#endif /*TIME_MEASUREMENT_H*/
