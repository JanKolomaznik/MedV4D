/**
 * @author Attila Ulman
 * @file MultiscanSegmentationFilter.h
 * @{ 
 **/

#ifndef MULTISCAN_SEGMENTATION_FILTER_H
#define MULTISCAN_SEGMENTATION_FILTER_H

#include "Imaging/AImageFilterWholeAtOnce.h"


namespace M4D {
namespace Imaging {

#define	BACKGROUND                  0

template< typename ImageType >
class MultiscanSegmentationFilter;

/**
 * Filter implementing multiscan, times series segmentation. 
 * Brain segmentation is done on the first slice of times series - region growing (non-recursive scanline floodfill). 
 * During this step binary mask is created which is used for other slices of the times series.
 */
template< typename ElementType >
class MultiscanSegmentationFilter< Image< ElementType, 3 > >
	: public AImageFilterWholeAtOnceIExtents< Image< ElementType, 3 >, Image< ElementType, 3 > >
{
  public:	

	  typedef Image< ElementType, 3 > InputImageType;
	  typedef Image< ElementType, 3 > OutputImageType;
	  typedef AImageFilterWholeAtOnceIExtents< InputImageType, OutputImageType > PredecessorType;

    /**
     * Properties structure.
     */
	  struct Properties: public PredecessorType::Properties
	  {
      /**
       * Properties constructor - fills up the Properties with default values.
       */
		  Properties ()
        : examinedSliceNum( EXEMINED_SLICE_NUM ), boneDensityBottom( BONE_DENSITY_BOTTOM ), background( BACKGROUND )
      {}

      /// Number of examined slices (number of time series).
		  uint32 examinedSliceNum;
      /// Lower bound of the interval of values (HU), which are considered as density of bones. 
      ElementType	boneDensityBottom;
      /// Background around segmented brain.
      ElementType	background;
	  };

    /**
     * Segmentation filter constructor.
     *
     * @param prop pointer to the properties structure
     */
	  MultiscanSegmentationFilter ( Properties *prop );

    /**
     * Segmentation filter constructor.
     */
	  MultiscanSegmentationFilter ();

    /**
     * Segmentation filter destructor.
     */
    ~MultiscanSegmentationFilter ();

	  GET_SET_PROPERTY_METHOD_MACRO(uint32, ExaminedSliceNum, examinedSliceNum);
    GET_SET_PROPERTY_METHOD_MACRO(ElementType, BoneDensityBottom, boneDensityBottom);
    GET_SET_PROPERTY_METHOD_MACRO(ElementType, Background, background);
  
  protected:

    /**
	   * The method executed by the pipeline's filter execution thread. The main functionality
     * of the filter is done by this method.
     *
	   *  @param in reference to the input image dataset
     *  @param out reference to the output image dataset
     *  @return true if finished successfully, false otherwise
     */
	  bool ProcessImage ( const InputImageType &in, OutputImageType &out );

  private:

	  GET_PROPERTIES_DEFINITION_MACRO;
};

} // namespace Imaging
} // namespace M4D


// include implementation
#include "MultiscanSegmentationFilter.tcc"

#endif // MULTISCAN_SEGMENTATION_FILTER_H

/** @} */
