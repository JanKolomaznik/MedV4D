#include "Log.h"
#include <iostream>
#include <iomanip>
#include <ctime>

std::ostream *logStream = &std::cout;

std::ostream &
operator<<( std::ostream & stream, const LogFormatter & logFormater )
{
	logFormater.Apply( stream );
	return stream;
}

void
LogDelimiter::Apply( std::ostream &stream ) const
{
	for ( unsigned i = 0; i < _length; ++i ) {
		stream << _delimiter;
	}
	stream << std::endl;
}

void
LogCurrentTime::Apply( std::ostream &stream ) const
{
	std::time_t date;    

	std::time( &date );

	stream << std::ctime( &date );
}
