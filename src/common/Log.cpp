#include "Log.h"
#include <iostream>
#include <iomanip>

std::ostream *logStream = &std::cout;

std::ostream &
operator<<( std::ostream & stream, const LogFormater & logFormater )
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
