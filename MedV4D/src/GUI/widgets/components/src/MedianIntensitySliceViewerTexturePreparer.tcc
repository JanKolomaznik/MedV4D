#ifndef MEDIAN_INTENSITY_SLICEVIEWER_TEXTURE_PREPARER_H
#error File MedianIntensitySliceViewerTexturePreparer.tcc cannot be included directly!
#else

namespace M4D
{
namespace Viewer
{

template< typename ElementType >
ElementType*
MedianIntensitySliceViewerTexturePreparer< ElementType >
::IntensityArranger(
        ElementType** channels,
        uint32 channelNumber,
        uint32 width,
        uint32 height )
    {
        ElementType* texture = new ElementType[ width * height ];
	uint32 num, realChannelNumber = 0;

	for ( uint32 k = 0; k < channelNumber; k++ )
	    if ( channels[k] ) realChannelNumber++;

	ElementType *sorter = new ElementType[ realChannelNumber ];

	// loop through all the pixels
        for ( uint32 i = 0; i < height; i++ )
            for ( uint32 j = 0; j < width; j++ )
            {

		num = 0;
                for ( uint32 k = 0; k < channelNumber; k++ )
                {
                    if ( channels[k] )
		    {
			sorter[ num++ ] = channels[k][ i * width + j ];
			if ( num >= realChannelNumber ) break;
		    }
                }

		// sort the intensity values of the input datasets at
		// the given position
		for ( int32 k = realChannelNumber - 1; k >= 0; k-- )
		    for ( int32 l = 0; l < k; l++ )
			if ( sorter[ l ] > sorter[ l + 1 ] )
			{
			    ElementType tmp = sorter[ l ];
			    sorter[ l ] = sorter[ l + 1 ];
			    sorter[ l + 1 ] = tmp;
			}

		// set the median of the sorted values as output
		texture[ i * width + j ] = sorter[ realChannelNumber / 2 ];
            }

	delete sorter;

        return texture;

    }

} /*namespace Viewer*/
} /*namespace M4D*/

#endif
