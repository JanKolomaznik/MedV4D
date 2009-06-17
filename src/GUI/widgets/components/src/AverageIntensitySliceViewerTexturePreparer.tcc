#ifndef AVERAGE_INTENSITY_SLICEVIEWER_TEXTURE_PREPARER_H
#error File AverageIntensitySliceViewerTexturePreparer.tcc cannot be included directly!
#else

namespace M4D
{
namespace Viewer
{

template< typename ElementType >
ElementType*
AverageIntensitySliceViewerTexturePreparer< ElementType >
::IntensityArranger(
        ElementType** channels,
        uint32 channelNumber,
        uint32 width,
        uint32 height )
    {
        ElementType* texture = new ElementType[ width * height ];
	uint32 realChannelNumber = 0;

	for ( uint32 k = 0; k < channelNumber; k++ )
	    if ( channels[k] ) realChannelNumber++;

        for ( uint32 i = 0; i < height; i++ )
            for ( uint32 j = 0; j < width; j++ )
            {
		texture[ i * width + j ] = 0;
                for ( uint32 k = 0; k < channelNumber; k++ )
                {
                    if ( channels[k] )
		    {
			texture[ i * width + j ] += ( channels[k][ i * width + j ] / realChannelNumber );
		    }
                }
            }

        return texture;

    }

} /*namespace Viewer*/
} /*namespace M4D*/

#endif
