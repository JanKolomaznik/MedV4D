#ifndef _REMOTE_FILTER_H
#define _REMOTE_FILTER_H

#include "Common.h"
#include "Imaging/AbstractImageFilterWholeAtOnce.h"
#include "cellBE/cellClient.h"

namespace M4D
{

namespace Imaging
{





template< typename InputImageType, typename OutputImageType >
class RemoteFilter 
	: public AbstractImageFilterWholeAtOnce< InputImageType, OutputImageType >
{
public:
	typedef typename  Imaging::AbstractImageFilterWholeAtOnce< InputImageType, OutputImageType > PredecessorType;
	struct Properties : public PredecessorType::Properties
	{

	};

	RemoteFilter();
	~RemoteFilter();
protected:

	bool
	ProcessImage(
			const InputImageType 	&in,
			OutputImageType		&out
		    );

	void
	PrepareOutputDatasets();
private:
	GET_PROPERTIES_DEFINITION_MACRO;

};

} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "Imaging/filters/RemoteFilter.tcc"

#endif /*_REMOTE_FILTER_H*/
