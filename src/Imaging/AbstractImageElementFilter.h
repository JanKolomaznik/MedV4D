#ifndef _ABSTRACT_IMAGE_ELEMENT_FILTER_H
#define _ABSTRACT_IMAGE_ELEMENT_FILTER_H

#include "Common.h"
#include "Imaging/AbstractImageSliceFilter.h"
#include "Imaging/AbstractImageFilterWholeAtOnce.h"
#include <vector>

namespace M4D
{

namespace Imaging
{



/**
 * We disallow general usage of template - only specializations.
 **/
template< typename InputImageType, typename OutputImageType, typename ElementFilter >
class AbstractImageElementFilter;


template< typename InputElementType, typename OutputElementType, typename ElementFilter >
class AbstractImageElementFilter< Image< InputElementType, 2 >, Image< OutputElementType, 2 >, ElementFilter >
	 : public AbstractImageFilterWholeAtOnceIExtents< Image< InputElementType, 2 >, Image< OutputElementType, 2 > >
{
public:
	typedef AbstractImageFilterWholeAtOnceIExtents< Image< InputElementType, 2 >, Image< OutputElementType, 2 > >	PredecessorType;
	
	struct Properties : public PredecessorType::Properties
	{
		Properties() {}

	};
	
	AbstractImageElementFilter( Properties *prop );
	~AbstractImageElementFilter() {}

protected:

	

	bool
	ProcessImage(
			const Image< InputElementType, 2 > 	&in,
			Image< OutputElementType, 2 >		&out
		    );


	ElementFilter	_elementFilter;
private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( AbstractImageElementFilter );
};


template< typename InputElementType, typename OutputElementType, typename ElementFilter >
class AbstractImageElementFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 >, ElementFilter >
	 : public IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
{
public:
	typedef typename Imaging::IdenticalExtentsImageSliceFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >	PredecessorType;
	
	struct Properties : public PredecessorType::Properties
	{
		Properties(): PredecessorType::Properties( 0, 10 ) {}

	};
	
	AbstractImageElementFilter( Properties *prop );
	~AbstractImageElementFilter() {}

protected:

	

	bool
	ProcessSlice(
			const Image< InputElementType, 3 > 	&in,
			Image< OutputElementType, 3 >		&out,
			int32					x1,
			int32					y1,
			int32					x2,
			int32					y2,
			int32					slice
		    );


	ElementFilter	_elementFilter;
private:
	/**
	 * Prohibition of copying.
	 **/
	PROHIBIT_COPYING_OF_OBJECT_MACRO( AbstractImageElementFilter );
};


} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "Imaging/AbstractImageElementFilter.tcc"

#endif /*_ABSTRACT_IMAGE_ELEMENT_FILTER_H*/
