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

template< typename ElementType >
class MultiscanSegmentationFilter< Image< ElementType, 3 > >
	: public AImageFilterWholeAtOnceIExtents< Image< ElementType, 3 >, Image< ElementType, 3 > >
{
  public:	

	  typedef Image< ElementType, 3 > InputImageType;
	  typedef Image< ElementType, 3 > OutputImageType;
	  typedef AImageFilterWholeAtOnceIExtents< InputImageType, OutputImageType > PredecessorType;

	  struct Properties: public PredecessorType::Properties
	  {
		  Properties ()
        : examinedSliceNum( EXEMINED_SLICE_NUM ), boneDensityBottom( BONE_DENSITY_BOTTOM ), background( BACKGROUND )
      {}

		  uint32 examinedSliceNum;
      ElementType	boneDensityBottom, background;
	  };

	  MultiscanSegmentationFilter ( Properties * prop );
	  MultiscanSegmentationFilter ();
    ~MultiscanSegmentationFilter ();

	  GET_SET_PROPERTY_METHOD_MACRO(uint32, ExaminedSliceNum, examinedSliceNum);
    GET_SET_PROPERTY_METHOD_MACRO(ElementType, BoneDensityBottom, boneDensityBottom);
    GET_SET_PROPERTY_METHOD_MACRO(ElementType, Background, background);
  
  protected:

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
