#ifndef BONE_SEGMENTATION_FILTER_H
#define BONE_SEGMENTATION_FILTER_H

#include "cellBE/RemoteFilter.h"

/**
 *  This is the firs exemplar of remote filter. It contains specification
 *  of an remote pipeline, that contains few filters that should perform
 *  bone segmentaion.
 *  This is example how remote filter currntly works and another remote
 *  filter should be written in this manner.
 *  Basic rule is to inherit from RemoteFilter templated class.
 */

// include needed filters ...
#include "Imaging/filters/ThresholdingFilter.h"

namespace M4D
{
namespace Imaging
{

template< typename ImageType >
class BoneSegmentationRemote
  : public RemoteFilter<ImageType, ImageType>
{
public:
	typedef typename RemoteFilter<ImageType, ImageType> PredecessorType;

	BoneSegmentationRemote();
  

	/////////////////// To customize /////////////////////
	// puting options available to outer world to be able to specify it ....
	typedef ThresholdingFilter<ImageType>	Thresholding;
	typedef typename Thresholding::Properties ThresholdingOptsType;
	
	ThresholdingOptsType *GetThreshholdingOptions( void)	
  {
		return &m_thresholdingOptions;
	}

protected:
	void PrepareOutputDatasets();

private:
	//GET_PROPERTIES_DEFINITION_MACRO;

	/**
	 * Here should be added members of  filter options type that will
	 * define the remote pipeline this filter represents. Each member
	 * for single filter in remote pipeline. As a next step is defining
	 * retrieving public members, that will provide ability to change
	 * the filter options from outer world.
	 **/
	ThresholdingOptsType m_thresholdingOptions;
	// ...


	//////////////////////////////////////////////////////

};

} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "cellBE/remoteFilters/BoneSegmentationRemote.tcc"

#endif
