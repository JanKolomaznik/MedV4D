#ifndef _EXCEPTION_BASE_H
#define _EXCEPTION_BASE_H

#include <exception>
#include <ostream>

namespace M4D
{
namespace ErrorHandling
{

class ExceptionBase: public std::exception
{
public:
	ExceptionBase( char * name );
	ExceptionBase();

	const char* what() { return _name; }	
protected:
	virtual void OnRaise();
private:
	char*	_name;
};

std::ostream& operator<<( std::ostream &out, ExceptionBase &exception );

} /*namespace ErrorHandling*/
} /*namespace M4D*/

#endif /*_EXCEPTION_BASE_H*/
