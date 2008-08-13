#include "Imaging/AbstractImageFilter.h"


namespace M4D
{
namespace Imaging
{

const AbstractImage&
GetInputImageFromPort( InputPortAbstractImage &port )
{
	return port.GetAbstractImage();
}

AbstractImage&
GetOutputImageFromPort( OutputPortAbstractImage &port )
{
	return port.GetAbstractImage();
}

}/*namespace Imaging*/
}/*namespace M4D*/
