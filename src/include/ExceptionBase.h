#ifndef _EXCEPTION_BASE_H
#define _EXCEPTION_BASE_H

#include <exception>
#include <ostream>
#include <string>

namespace M4D
{
namespace ErrorHandling
{

class ExceptionBase: public std::exception
{
public:
	ExceptionBase( std::string name ) throw();
	ExceptionBase() throw();
	~ExceptionBase() throw(){}
	const char* what() const  throw(){ return _name.data(); }	
protected:
	virtual void OnRaise();
private:
	std::string	_name;
};

class ExceptionWrongPointer: public ExceptionBase
{
public:
	ExceptionWrongPointer( std::string, void *pointer ) throw();
private:
	void	*_pointer;
};

std::ostream& operator<<( std::ostream &out, ExceptionBase &exception );

} /*namespace ErrorHandling*/
} /*namespace M4D*/

#endif /*_EXCEPTION_BASE_H*/
