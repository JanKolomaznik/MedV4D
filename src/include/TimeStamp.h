#ifndef _TIME_STAMP_H
#define _TIME_STAMP_H

#include "Common.h"

namespace M4D
{
namespace Common
{

class TimeStamp
{
public:
	TimeStamp();

	TimeStamp( const TimeStamp& b );

	~TimeStamp();

	const TimeStamp&
	operator=( const TimeStamp& b );

	void
	Increase();

	bool
	IdenticalID( const TimeStamp& b );

	const TimeStamp&
	operator++();

	const TimeStamp
	operator++(int);

	bool
	operator<( const TimeStamp& b );

	bool
	operator<=( const TimeStamp& b );
private:
	static uint64
	GenerateUniqueID();

	uint64 _uniqueID;
	long _timeStamp;

};


}/*namespace Common*/
}/*namespace M4D*/

#endif /*_TIME_STAMP_H*/
