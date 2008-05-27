#ifndef _CONNECTION_H
#define _CONNECTION_H

#include "Imaging/AbstractFilter.h"
#include "Imaging/Ports.h"
#include <boost/shared_ptr.hpp>

namespace M4D
{
namespace Imaging
{

class Connection
{
public:
	typedef boost::shared_ptr< Connection > ConnectionPtr;

	virtual 
	~Connection() {}

	virtual void
	ConnectIn( OutputPort& outputPort )=0;

	virtual void
	ConnectOut( InputPort& inputPort )=0;

	virtual void
	DisconnectInFilter()=0;

	virtual void
	DisconnectOutFilter( InputPort& inputPort )=0;

protected:

private:
	Connection( const Connection& );
	Connection&
	operator=( const Connection& );
};

}/*namespace Imaging*/
}/*namespace M4D*/


#endif /*_CONNECTION_H*/
