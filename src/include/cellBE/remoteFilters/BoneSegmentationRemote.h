#ifndef BONE_SEGMENTATION_FILTER_H
#define BONE_SEGMENTATION_FILTER_H

// include needed filters ...
#include "Imaging/filters/ThresholdingFilter.h"

namespace M4D
{

namespace Imaging
{

template< class InType, class OutType>
class BoneSegmentationRemote
  : public RemoteFilter<InType, OutType>
{
private:
	GET_PROPERTIES_DEFINITION_MACRO;

  /**
   *  Here should be added members of  filter options type that will
   *  define the remote pipeline this filter represents. Each member
   *  for single filter in remote pipeline. As a next step is defining
   *  retrieving public members, that will provide ability to change
   *  the filter options from outer world.
   */
  ThresholdingFilterMaskOptions<InType> m_thresholdingOptions;
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

public:
  BoneSegmentationRemote();


  /////////////////// To customize /////////////////////
  // puting options available to outer world to be able to specify it ....
  typedef ThresholdingFilterMaskOptions<InType> ThresholdingOptsType;
  inline ThresholdingOptsType *GetThreshholdingOptions( void)
  {
    return &m_thresholdingOptions;
  }

  //////////////////////////////////////////////////////

};

} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "cellBE/remoteFilters/BoneSegmentationRemote.tcc"

#endif
