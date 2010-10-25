#ifndef _IMAGE_TOOLS_H
#define _IMAGE_TOOLS_H

#include "Imaging/ImageRegion.h"
#include "Imaging/Histogram.h"

namespace M4D {
namespace Imaging {

template< typename TImageRegion, typename THistogram >
void
AddRegionToHistogram( THistogram &aHistogram, const TImageRegion &aRegion )
{
	//TODO - test TImageRegion type
	int32 min = aHistogram.GetMin() - 1;
	int32 max = aHistogram.GetMax();
	TImageRegion::Iterator it = aRegion.GetIterator();
	while ( !it.IsEnd() ) {
		aHistogram.FastIncCell( ClampToInterval( min, max, (int32)*it ) );
		++it;
	}
}



} /*namespace Imaging*/
} /*namespace M4D*/

#endif /*_IMAGE_TOOLS_H*/
