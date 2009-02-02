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

	std::string
	GetFileName()
	{
#ifdef DEBUG_LEVEL
		return _exceptionFileName;
#else
		return "";
#endif /*DEBUG_LEVEL*/
	}

	int
	GetLineNumber()
	{
#ifdef DEBUG_LEVEL
		return _exceptionLineNumber;
#else
		return -1;
#endif /*DEBUG_LEVEL*/
	}
protected:
	virtual void OnRaise();
private:
	std::string	_name;

#ifdef DEBUG_LEVEL
	std::string	_exceptionFileName;
	int		_exceptionLineNumber;
#endif /*DEBUG_LEVEL*/
};

class ExceptionCastProblem : public ExceptionBase
{
public:
	ExceptionCastProblem( std::string name ) throw() : ExceptionBase( name ) {}
	ExceptionCastProblem() throw() : ExceptionBase( "Problem with casting objects to another type" ) {}
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

class EBadDimension: public ExceptionBase
{
public:
	EBadDimension( std::string name ) throw(): ExceptionBase( name ) {}
	EBadDimension() throw(): ExceptionBase( "Wrong dimension" ) {}
private:

};

class EBadIndex: public ExceptionBase
{
public:
	EBadIndex( std::string name ) throw(): ExceptionBase( name ) {}
	EBadIndex() throw(): ExceptionBase( "Wrong index" ) {}
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
