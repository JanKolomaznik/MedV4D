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
	Connection() {}

	virtual void
	ConnectOutFilter( AbstractFilter::AbstractFilterPtr filter )=0;

	virtual void
	ConnectInFilter( AbstractFilter::AbstractFilterPtr filter )=0;

	virtual void
	DisconnectInFilter()=0;

	virtual void
	DisconnectOutFilter( AbstractFilter::AbstractFilterPtr filter )=0;

protected:

private:

};

}/*namespace Imaging*/
}/*namespace M4D*/


#endif /*_CONNECTION_H*/
