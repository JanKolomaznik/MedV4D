#ifndef _PORTS_H
#define _PORTS_H

#include <boost/shared_ptr.hpp>

namespace M4D
{
namespace Imaging
{

class Port
{
public:
	virtual
	~Port() {}

protected:

private:

};

/**
 *
 **/
class InputPort
{
public:
	virtual
	~InputPort() {}

protected:

private:

};

/**
 *
 **/
class OutputPort
{
public:
	virtual
	~OutputPort() {}

protected:

private:

};

//******************************************************************************
template< typename ElementType >
class Input2DImageFilter
{

};

template< typename ElementType >
class Input3DImageFilter
{

};

template< typename ElementType >
class Output2DImageFilter
{

};

template< typename ElementType >
class Output3DImageFilter
{

};
//******************************************************************************

class InputPortList
{
private:
	//Not implemented
	InputPortList( const InputPortList& );
	InputPortList&
	operator=( const InputPortList& );
};

class OutputPortList
{
private:
	//Not implemented
	OutputPortList( const OutputPortList& );
	OutputPortList&
	operator=( const OutputPortList& );
};

//******************************************************************************

}/*namespace Imaging*/
}/*namespace M4D*/


#endif /*_PORTS_H*/

