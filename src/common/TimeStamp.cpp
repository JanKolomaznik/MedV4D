#include "TimeStamp.h"
#include "Thread.h"

namespace M4D
{
namespace Common
{

//TODO locking

TimeStamp::TimeStamp()
: _uniqueID( TimeStamp::GenerateUniqueID() ), _timeStamp( 1 )
{
	
}

TimeStamp::TimeStamp( const TimeStamp& b )
	: _uniqueID( b._uniqueID ), _timeStamp( b._timeStamp )
{

}

TimeStamp::~TimeStamp()
{

}

const TimeStamp&
TimeStamp::operator=( const TimeStamp& b )
{
	_uniqueID = b._uniqueID;
	_timeStamp = b._timeStamp;
	return *this;
}

void
TimeStamp::Increase()
{
	operator++();
}

bool
TimeStamp::IdenticalID( const TimeStamp &b )
{
	return this->_uniqueID == b._uniqueID;
}

const TimeStamp&
TimeStamp::operator++()
{
	++(this->_timeStamp);
	return *this;
}

const TimeStamp
TimeStamp::operator++(int)
{
	TimeStamp copy = *this;
	
	++(this->_timeStamp);
	
	return copy;
}

bool
TimeStamp::operator<( const TimeStamp& b )
{
	return this->_timeStamp < b._timeStamp;
}

bool
TimeStamp::operator<=( const TimeStamp& b )
{
	return this->_timeStamp <= b._timeStamp;
}

uint64
TimeStamp::GenerateUniqueID()
{
	static uint64 lastID = 0;
	static Multithreading::Mutex genMutex;

	{	//We must synchronize to avoid multiple generation of one ID.
		Multithreading::ScopedLock lock( genMutex );
		return ++lastID;
	}
}

}/*namespace Common*/
}/*namespace M4D*/
