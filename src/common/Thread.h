#ifndef _THREAD_H
#define _THREAD_H

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/thread/barrier.hpp>

#undef GetMessage
#undef SendMessage

/**
 *  @ingroup common
 *  @file Thread.h
 *
 *  @addtogroup common
 *  @{
 *  @section threading Threading support
 *
 *  This is only header with wrappers to boost::thread classes (mutexes, ...). 
 *  Only purpose is to provide interace for multithreading with option to
 *  change multithreading framework.
 */

namespace M4D
{
namespace Multithreading
{

typedef boost::thread	Thread;

typedef boost::mutex	Mutex;

typedef boost::mutex::scoped_lock	ScopedLock;

typedef boost::recursive_mutex	RecursiveMutex;

typedef boost::recursive_mutex::scoped_lock	RecursiveScopedLock;

typedef boost::barrier Barrier;

void
yield();

void
sleep( int duration );

}/*namespace Multithreading*/
}/*namespace M4D*/

/** @} */

#endif /*_THREAD_H*/

