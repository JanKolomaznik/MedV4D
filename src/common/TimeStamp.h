#ifndef _TIME_STAMP_H
#define _TIME_STAMP_H

#include "Common.h"
#include "Thread.h"

/**
 *  @ingroup common
 *  @file TimeStamp.h
 *
 *  @addtogroup common
 *  @{
 *  @section timestaps TimeStamps
 *  
 *  Very important class TimeStamp, which is used every time when change
 *  detection is needed. Its implemented thread safe, so it can be 
 *	used even for synchronization purposes.
 */

namespace M4D
{
namespace Common
{


//TODO mutable
class TimeStamp
{
public:
	TimeStamp();

	TimeStamp( const TimeStamp& b );

	~TimeStamp();

	TimeStamp
	operator=( const TimeStamp& b );

	void
	Increase();

	bool
	IdenticalID( const TimeStamp& b );

	TimeStamp
	operator++();

	TimeStamp
	operator++(int);

	bool
	operator>( const TimeStamp& b )const;

	bool
	operator>=( const TimeStamp& b )const;

	bool
	operator==( const TimeStamp& b )const;

	bool
	operator!=( const TimeStamp& b )const;
private:
	static uint64
	GenerateUniqueID();

	uint64 _uniqueID;
	long _timeStamp;

	Multithreading::Mutex	_accessLock;
};

extern const TimeStamp	DefaultTimeStamp;

}/*namespace Common*/
}/*namespace M4D*/

/** @} */

#endif /*_TIME_STAMP_H*/

