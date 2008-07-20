#ifndef _THREAD_H
#define _THREAD_H

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/thread/xtime.hpp>


namespace M4D
{
namespace Multithreading
{

typedef boost::thread	Thread;

typedef boost::mutex	Mutex;

typedef boost::mutex::scoped_lock	ScopedLock;

typedef boost::recursive_mutex	RecursiveMutex;

typedef boost::recursive_mutex::scoped_lock	RecursiveScopedLock;

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


#endif /*_THREAD_H*/

