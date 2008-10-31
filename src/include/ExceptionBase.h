#ifndef _EXCEPTION_BASE_H
#define _EXCEPTION_BASE_H

#include <exception>
#include <ostream>
#include <string>

/**
 *  @ingroup common
 *  @file ExceptionBase.h
 *
 *  @addtogroup common
 *  @{
 *  @section exceptionhandling Exception handling
 *
 *  Defines base class that is used for exception handling. All other
 *  exceptions within the project should inherit from her.
 */

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
	//TODO
private:
	void	*_pointer;
};

class EWrongDimension: public ExceptionBase
{
public:
	EWrongDimension( std::string name ) throw(): ExceptionBase( name ) {}
	EWrongDimension() throw(): ExceptionBase( "Wrong dimension" ) {}
private:

};

class EWrongIndex: public ExceptionBase
{
public:
	EWrongIndex( std::string name ) throw(): ExceptionBase( name ) {}
	EWrongIndex() throw(): ExceptionBase( "Wrong index" ) {}
private:

};

class ETODO: public ExceptionBase
{
public:
	ETODO( std::string name ) throw(): ExceptionBase( name ) {}
	ETODO() throw(): ExceptionBase( "TODO - not finished." ) {}
private:

};

std::ostream& operator<<( std::ostream &out, ExceptionBase &exception );

} /*namespace ErrorHandling*/
} /*namespace M4D*/

/** @} */

#endif /*_EXCEPTION_BASE_H*/
