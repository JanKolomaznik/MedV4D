#ifndef BONE_SEGMENTATION_FILTER_H
#define BONE_SEGMENTATION_FILTER_H

#include "cellBE/RemoteFilter.h"

// include needed filters ...
#include "Imaging/filters/ThresholdingFilter.h"

namespace M4D
{
namespace Imaging
{

template< typename InType, typename OutType>
class BoneSegmentationRemote
  : public RemoteFilter<InType, OutType>
{
public:
	typedef typename RemoteFilter<InType, OutType> PredecessorType;

	BoneSegmentationRemote();


	/////////////////// To customize /////////////////////
	// puting options available to outer world to be able to specify it ....
	typedef typename ThresholdingFilterMaskOptions<InType> ThresholdingOptsType;
	
	ThresholdingOptsType *GetThreshholdingOptions( void)
		{
			return &m_thresholdingOptions;
		}

private:
	GET_PROPERTIES_DEFINITION_MACRO;

	/**
	 * Here should be added members of  filter options type that will
	 * define the remote pipeline this filter represents. Each member
	 * for single filter in remote pipeline. As a next step is defining
	 * retrieving public members, that will provide ability to change
	 * the filter options from outer world.
	 **/
	typename ThresholdingFilterMask<InType>::Properties m_thresholdingOptions;
	// ...

	ClientJob *m_job;

protected:

	//bool
	//ProcessImage(
	//		const InputImageType 	&in,
	//		OutputImageType		&out
	//	    );

	//void
	//PrepareOutputDatasets();


	//////////////////////////////////////////////////////

};

} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "cellBE/remoteFilters/BoneSegmentationRemote.tcc"

#endif
