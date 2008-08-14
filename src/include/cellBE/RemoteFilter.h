#ifndef _REMOTE_FILTER_H
#define _REMOTE_FILTER_H

#include "Imaging/AbstractImageFilterWholeAtOnce.h"
#include "cellBE/RemoteFilterBase.h"

namespace M4D
{

namespace Imaging
{

template< typename InputImageType, typename OutputImageType >
class RemoteFilter 
	: public AbstractImageFilterWholeAtOnce< InputImageType, OutputImageType >
  , public M4D::CellBE::RemoteFilterBase
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

  /**
   *  This method should count output image size based on filters
   *  that are in remote pipeline through SetImageSize() method
   *  of output ports.
   */
	//virtual void PrepareOutputDatasets() = 0;

  // actual job responsible for actual work
  M4D::CellBE::ClientJob *m_job;

private:
	GET_PROPERTIES_DEFINITION_MACRO;

};

} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "cellBE/RemoteFilter.tcc"

#endif /*_REMOTE_FILTER_H*/
