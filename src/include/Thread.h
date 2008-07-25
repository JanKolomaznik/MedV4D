#ifndef _THREAD_H
#define _THREAD_H

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/recursive_mutex.hpp>

#undef AddPort
#undef GetMessage
#undef SendMessage



namespace M4D
{
namespace Multithreading
{

typedef boost::thread	Thread;

typedef boost::mutex	Mutex;

typedef boost::mutex::scoped_lock	ScopedLock;

typedef boost::recursive_mutex	RecursiveMutex;

typedef boost::recursive_mutex::scoped_lock	RecursiveScopedLock;

void
yield();

void
sleep( int duration );

}/*namespace Multithreading*/
}/*namespace M4D*/


#endif /*_THREAD_H*/

