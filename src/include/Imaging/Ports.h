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
template< ElementType >
class Input2DImageFilter
{

};

template< ElementType >
class Input3DImageFilter
{

};

template< ElementType >
class Output2DImageFilter
{

};

template< ElementType >
class Output3DImageFilter
{

};
//******************************************************************************

class InputPortList
{

};

class OutputPortList
{

};

//******************************************************************************

}/*namespace Imaging*/
}/*namespace M4D*/


#endif /*_PORTS_H*/

