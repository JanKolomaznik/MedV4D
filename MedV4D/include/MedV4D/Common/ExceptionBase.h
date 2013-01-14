#ifndef _EXCEPTION_BASE_H
#define _EXCEPTION_BASE_H

#include <exception>
#include <ostream>
#include <string>
#include "MedV4D/Common/StringTools.h"
#include "MedV4D/Common/Types.h"
#include <boost/exception/all.hpp>
#include <boost/filesystem/path.hpp>
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

typedef ExceptionCastProblem ECastProblem;

class ENULLPointer : public ExceptionBase
{
public:
	ENULLPointer( std::string name ) throw() : ExceptionBase( name ) {}
	ENULLPointer() throw() : ExceptionBase( "Passed NULL pointer." ) {}
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

class EAllocationFailed: public ExceptionBase
{
public:
	EAllocationFailed( std::string name ) throw() : ExceptionBase( name ) {}
	EAllocationFailed() throw() : ExceptionBase( "Allocation failed" ) {}
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
	EBadIndex( std::string name) throw(): ExceptionBase( name ), _idx( -1 ) {}
	EBadIndex( std::string name, int32 idx ) throw(): ExceptionBase( TO_STRING( name << ": " << idx ) ), _idx( idx ) {}
	EBadIndex() throw(): ExceptionBase( "Wrong index" ), _idx( -1 ) {}

	int32
	GetIndex() const
	{ return _idx; }
private:
	int32	_idx;

};

class EBadParameter: public ExceptionBase
{
public:
	EBadParameter( std::string name ) throw(): ExceptionBase( name ) {}
	EBadParameter() throw(): ExceptionBase( "Bad parameter" ) {}
private:

};

class ENotFound: public ExceptionBase
{
public:
	ENotFound( std::string name ) throw(): ExceptionBase( name ) {}
	ENotFound() throw(): ExceptionBase( "Not found." ) {}
private:

};

class EAlreadyPresent: public ExceptionBase
{
public:
	EAlreadyPresent( std::string name ) throw(): ExceptionBase( name ) {}
	EAlreadyPresent() throw(): ExceptionBase( "Already present." ) {}
private:

};

class EObjectUnavailable: public ExceptionBase
{
public:
	EObjectUnavailable( std::string name ) throw(): ExceptionBase( name ) {}
	EObjectUnavailable() throw(): ExceptionBase( "Object is not available." ) {}
private:

};

class ETODO: public ExceptionBase
{
public:
	ETODO( std::string name ) throw(): ExceptionBase( name ) {}
	ETODO() throw(): ExceptionBase( "TODO - not finished." ) {}
private:

};

class EShouldntReachThis: public ExceptionBase
{
public:
	EShouldntReachThis() throw(): ExceptionBase( "Shouldn't reach this." ) {}
private:

};

class EFileProblem : public ExceptionBase
{
public:
	EFileProblem( std::string name, Path fileName ) throw() : ExceptionBase( name ), _fileName( fileName ) {}
	EFileProblem( Path fileName ) throw() : ExceptionBase( TO_STRING( "Problem with file" << fileName ) ), _fileName( fileName ) {}
	~EFileProblem() throw(){}

	const Path &
	Filename() const
	{ return _fileName; }

protected:
	Path _fileName;
};

class EFileNotFound : public EFileProblem
{
public:
	EFileNotFound( Path fileName ) throw() : EFileProblem( TO_STRING( "File not found: " << fileName ), fileName ) {}
	~EFileNotFound() throw(){}

protected:
};


class EInitError: public ExceptionBase
{
public:
	EInitError( std::string name ) throw(): ExceptionBase( TO_STRING( "Initialization error :" << name ) ) {}
private:

};

std::ostream& operator<<( std::ostream &out, ExceptionBase &exception );


struct exception_base: virtual std::exception, virtual boost::exception { };
struct io_error: virtual exception_base { };
struct EDirNotFound: virtual io_error { };
//struct EFileNotFound: virtual io_error { };

struct EItemNotFound: virtual exception_base { };
struct EObjectNotInitialized: virtual exception_base { };

//error infos
typedef boost::error_info<struct tag_Path, boost::filesystem::path> EInfoPath;
typedef boost::error_info<struct tag_wMessage, std::wstring> EInfoWMessage;
typedef boost::error_info<struct tag_Message, std::string> EInfoMessage;
typedef boost::error_info<struct tag_ArgumentName, std::string> EInfoArgumentName;
typedef boost::error_info<struct tag_ItemName, std::string> EInfoItemName;
typedef boost::error_info<struct tag_ItemIndex, size_t> EInfoItemIndex;
typedef boost::error_info<struct tag_ObjectName, std::string> EInfoObjectName;

} /*namespace ErrorHandling*/
} /*namespace M4D*/

/** @} */

#endif /*_EXCEPTION_BASE_H*/
