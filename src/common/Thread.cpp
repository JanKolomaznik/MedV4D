/**
 *  @ingroup common
 *  @file Thread.cpp
 *  @author Jan Kolomaznik
 */
#include "Thread.h"

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

void
sleep( int duration )
{
	boost::xtime sleepTime;
	boost::xtime_get(&sleepTime, boost::TIME_UTC);
	sleepTime.nsec += duration;
	boost::thread::sleep(sleepTime);
}

}/*namespace Multithreading*/
}/*namespace M4D*/
