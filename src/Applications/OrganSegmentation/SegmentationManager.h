#ifndef _SEGMENTATION_MANAGER_H
#define _SEGMENTATION_MANAGER_H

#include "Imaging/Image.h"

enum SegmentationType
{
	stManual,
	

};


class SegmentationManager
{
public:
	

	void
	SetInputImage( M4D::Imaging::AImage::AImagePtr image );

	void
	SetSegmentationType( SegmentationType sType );

	void
	KidneyInitializationPoints( ... points );

};


#endif /*_SEGMENTATION_MANAGER_H*/


