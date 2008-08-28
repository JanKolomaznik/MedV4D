/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file BoneSegmentationRemote.h 
 * @{ 
 **/

#ifndef BONE_SEGMENTATION_FILTER_H
#define BONE_SEGMENTATION_FILTER_H

#include "cellBE/RemoteFilter.h"


// include needed filters ...
#include "Imaging/filters/ThresholdingFilter.h"
#include "Imaging/filters/MedianFilter.h"

namespace M4D
{
namespace Imaging
{

/**
 *  This is the firs exemplar of remote filter. It contains specification
 *  of an remote pipeline, that contains few filters that should perform
 *  bone segmentaion.
 *  This is example how remote filter currntly works and another remote
 *  filter should be written in this manner.
 *  Basic rule is to inherit from RemoteFilter templated class.
 */
template< typename ImageType >
class BoneSegmentationRemote
  : public RemoteFilter<ImageType, ImageType>
{
public:
	typedef typename RemoteFilter<ImageType, ImageType> PredecessorType;
	typedef PredecessorType::Properties	Properties;

	BoneSegmentationRemote();
  

	/////////////////// To customize /////////////////////
	// puting options available to outer world to be able to specify it ....

  // thresholding filter issues
	typedef ThresholdingFilter<ImageType>	Thresholding;
	typedef typename Thresholding::Properties ThresholdingOptsType;
	
	ThresholdingOptsType *GetThreshholdingOptions( void)	
  {
		return &m_thresholdingOptions;
	}

  // median filter issues
  typedef MedianFilter2D<ImageType>	Median;
	typedef typename Median::Properties MedianOptsType;
	
	MedianOptsType *GetMedianOptions( void)	
  {
		return &m_medianOptions;
	}

protected:
	void PrepareOutputDatasets();

private:

	/**
	 * Here should be added members of  filter options type that will
	 * define the remote pipeline this filter represents. Each member
	 * for single filter in remote pipeline. As a next step is defining
	 * retrieving public members, that will provide ability to change
	 * the filter options from outer world.
	 **/
	ThresholdingOptsType m_thresholdingOptions;
  MedianOptsType m_medianOptions;
	// ...


	//////////////////////////////////////////////////////

};

} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "cellBE/remoteFilters/BoneSegmentationRemote.tcc"

#endif

/** @} */

