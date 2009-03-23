/**
 *  @ingroup common
 *  @file TimeStamp.cpp
 *  @author Jan Kolomaznik
 */
#include "common/TimeStamp.h"
#include "common/Thread.h"

namespace M4D
{
namespace Common
{

const TimeStamp	DefaultTimeStamp;

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

TimeStamp
TimeStamp::operator=( const TimeStamp& b )
{
	Multithreading::ScopedLock lock( _accessLock );

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
	Multithreading::ScopedLock lock( _accessLock );

	return this->_uniqueID == b._uniqueID;
}

TimeStamp
TimeStamp::operator++()
{
	Multithreading::ScopedLock lock( _accessLock );

	++(this->_timeStamp);
	return *this;
}

TimeStamp
TimeStamp::operator++(int)
{
	//TODO check locking
	TimeStamp copy = *this;
	
	++(this->_timeStamp);
	
	return copy;
}

bool
TimeStamp::operator>( const TimeStamp& b )const
{
	//TODO - check if locking needed
	return this->_timeStamp > b._timeStamp;
}

bool
TimeStamp::operator>=( const TimeStamp& b )const
{

	//TODO - check if locking needed
	return this->_timeStamp >= b._timeStamp;
}

bool
TimeStamp::operator==( const TimeStamp& b )const
{

	//TODO - check if locking needed
	return this->_timeStamp == b._timeStamp;
}

bool
TimeStamp::operator!=( const TimeStamp& b )const
{

	//TODO - check if locking needed
	return this->_timeStamp != b._timeStamp;
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
