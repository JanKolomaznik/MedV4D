#include "Imaging/Pipeline.h"
#include <algorithm>

namespace M4D
{
namespace Imaging
{

Pipeline()
{

}

~Pipeline()
{
	std::for_each(
		_connections.begin(), 
		_connections.end(), 
		M4D::Functors::Deletor< Connection* > 
		);

	std::for_each(
		_filters.begin(), 
		_filters.end(), 
		M4D::Functors::Deletor< AbstractPipeFilter* > 
		);
}

void
Pipeline::AddFilter( AbstractPipeFilter *filter )
{
	if( filter == NULL ) {
		//TODO throw exception
	}
	_filters.push_back( filter );
}

void
Pipeline::FillingFinished()
{

}

Connection &
Pipeline::MakeConnection( OutputPort& outPort, InputPort& inPort )
{
	//if inPort occupied -error. Connection concept is designed one to many.
	if( inPort.IsPlugged() ) {
		//TODO throw exception
	}

	//if outPort is connected, we use already created Conncetion, otherwise we 
	//have to create new one.
	if( outPort.IsPlugged() ) {

	} else {

	}
}


}/*namespace Imaging*/
}/*namespace M4D*/

