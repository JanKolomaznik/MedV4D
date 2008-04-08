#ifndef _EXCEPTION_BASE_H
#define _EXCEPTION_BASE_H

#include <exception>
#include <ostream>

namespace ErrorHandling
{

class ExceptionBase: public std::exception
{
public:
	ExceptionBase( char * name );

	const char* what() { return _name; }	
private:
	char*	_name;
};

std::ostream& operator<<( std::ostream &out, ExceptionBase &exception );

} /*namespace ErrorHandling*/

#endif /*_EXCEPTION_BASE_H*/
