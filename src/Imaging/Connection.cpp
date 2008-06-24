#include "Imaging/Connection.h"

namespace M4D
{
namespace Imaging
{

//*****************************************************************************

void
AbstractImageConnection::PushConsumer( InputPortAbstractImage& consumer )
{
	if( _consumers.find( consumer.GetID() )== _consumers.end() ) {
		consumer.Plug( *this );
		_consumers[ consumer.GetID() ] = &consumer;
	} else {
		//TODO throw exception
	}
}

//*****************************************************************************

}/*namespace Imaging*/
}/*namespace M4D*/

