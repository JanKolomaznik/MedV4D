#ifndef _PIPELINE_H
#define _PIPELINE_H


#include "Imaging/AbstractPipeline.h"
#include "Imaging/ConnectionInterface.h"
#include <boost/shared_ptr.hpp>
#include <vector>

namespace M4D
{
namespace Imaging
{

class Pipeline : public AbstractPipeline
{
public:
	
	Pipeline();

	~Pipeline();

	void
	AddFilter( AbstractPipeFilter *filter );

	void
	FillingFinished();

	/**
	 * Connect two compatible ports if possible.
	 * @param outPort Reference to output port of some filter.
	 * @param inPort Reference to input port of some filter.
	 **/
	ConnectionInterface &
	MakeConnection( M4D::Imaging::OutputPort& outPort, M4D::Imaging::InputPort& inPort );
protected:
	typedef std::vector< AbstractPipeFilter * > FilterVector;
	typedef std::vector< ConnectionInterface * > ConnectionVector;

	FilterVector		_filters;
	ConnectionVector	_connections;

private:

};

}/*namespace Imaging*/
}/*namespace M4D*/


#endif /*_PIPELINE_H*/

