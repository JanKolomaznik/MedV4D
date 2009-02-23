/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file filterIDEnums.h 
 * @{ 
 **/

#ifndef FILTERID_ENUMS_H
#define FILTERID_ENUMS_H

/** 
 *	filter identification defines. Here are to be added new ones when
 *	new filter is written
 */
enum FilterID {
  FID_Thresholding,
  FID_Median,
  FID_SimpleProjection,
  FID_LevelSetSegmentation
};

#endif


/** @} */

