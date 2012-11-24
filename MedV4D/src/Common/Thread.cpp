/**
 *  @ingroup common
 *  @file Thread.cpp
 *  @author Jan Kolomaznik
 */
#include "MedV4D/Common/Thread.h"

#include <boost/thread/xtime.hpp>


namespace M4D
{
namespace Multithreading
{

void
yield()
{
	boost::thread::yield();
}

/*void
sleep( int duration )
{
	boost::xtime sleepTime;
	boost::xtime_get(&sleepTime, boost::TIME_UTC);
	sleepTime.nsec += duration;
	//boost::this_thread::sleep(boost::posix_time::milliseconds(_msecs));
	boost::thread::sleep(sleepTime);
}*/

}/*namespace Multithreading*/
}/*namespace M4D*/
