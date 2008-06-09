#ifndef _TIME_STAMP_H
#define _TIME_STAMP_H

#include "Common.h"
#include "Thread.h"

namespace M4D
{
namespace Common
{

class TimeStamp
{
public:
	TimeStamp();

	TimeStamp( const TimeStamp& );

	~TimeStamp();

	const TimeStamp&
	operator=( const TimeStamp& );

	void
	Increase();

	bool
	IdenticalID( const TimeStamp );

	const TimeStamp&
	operator++();

	const TimeStamp&
	operator++(int);

	bool
	operator<( const TimeStamp& b );

	bool
	operator<=( const TimeStamp& b );
private:
	uint64
	GenerateUniqueID();

	uint64 _uniqueID;
	long _timeStamp;

};


}/*namespace Common*/
}/*namespace M4D*/

#endif /*_TIME_STAMP_H*/
