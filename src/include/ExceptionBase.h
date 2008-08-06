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

class ExceptionCastProblem : public ExceptionBase
{
public:
	ExceptionCastProblem( std::string name ) throw() : ExceptionBase( name ) {}
	~ExceptionCastProblem() throw(){}
};

class ENULLPointer : public ExceptionBase
{
public:
	ENULLPointer( std::string name ) throw() : ExceptionBase( name ) {}
	ENULLPointer() throw() : ExceptionBase( "Obtained NULL pointer." ) {}
	~ENULLPointer() throw(){}
};

template< typename ParamType >
class ExceptionBadParameter : public ExceptionBase
{
public:
	ExceptionBadParameter( ParamType param ) throw() : ExceptionBase( "Bad parameter value." ), _param( param ) {}
	ExceptionBadParameter( ParamType param, std::string name ) throw() : ExceptionBase( name ), _param( param ) {}
	~ExceptionBadParameter() throw(){}

	ParamType
	GetParamValue()const
		{ return _param; }
protected:
	ParamType	_param;
};

class ENotFinished: public ExceptionBase
{
public:
	ENotFinished( std::string name ) throw() : ExceptionBase( name ) {}
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
