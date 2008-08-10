#ifndef FILTERID_ENUMS_H
#define FILTERID_ENUMS_H

#include "Imaging/filters/ThresholdingFilter.h"

/** 
 *	filter identification defines. Here are to be added new ones when
 *	new filter is written
 */
enum FilterID {
  Thresholding,
};

//*******************************************************
template< typename PropertiesType >
FilterID
GetFilterID( PropertiesType & prop );
//*******************************************************

template< typename InputImageType >
FilterID
GetFilterID< M4D::Imaging::ThresholdingFilter< InputImageType >::Properties >
	( typename M4D::Imaging::ThresholdingFilter< InputImageType >::Properties &prop )
{ 
  return Thresholding; 
}

//*******************************************************
/*template< typename PropertiesType >
void
AddToVector( Vector &v, PropertiesType *prop )
{
	v.insert( GetFilterID< PropertiesType >( *prop ), prop );
}*/
#endif

