#include "Thread.h"

#include <boost/thread/xtime.hpp>


namespace M4D
{
namespace Multithreading
{

inline void
yield()
{
	boost::thread::yield();
}

inline void
sleep( int duration )
{
	boost::xtime sleepTime;
	boost::xtime_get(&sleepTime, boost::TIME_UTC);
	sleepTime.nsec += duration;
	boost::thread::sleep(sleepTime);
}

}/*namespace Multithreading*/
}/*namespace M4D*/
