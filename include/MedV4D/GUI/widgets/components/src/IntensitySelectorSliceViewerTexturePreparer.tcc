#ifndef INTENSITY_SELECTOR_SLICEVIEWER_TEXTURE_PREPARER_H
#error File IntensitySelectorSliceViewerTexturePreparer.tcc cannot be included directly!
#else

namespace M4D
{
namespace Viewer
{

template< typename ElementType >
ElementType*
IntensitySelectorSliceViewerTexturePreparer< ElementType >
::IntensityArranger(
        ElementType** channels,
        uint32 channelNumber,
        uint32 width,
        uint32 height )
    {
        ElementType* texture = new ElementType[ width * height ];

        bool pixelSet = false;

	// loop through the input datasets for each position
        for ( uint32 i = 0; i < height; i++ )
            for ( uint32 j = 0; j < width; j++ )
            {
                pixelSet = false;
                texture[ i * width + j ] = TypeTraits< ElementType >::Min;
                for ( uint32 k = 0; k < channelNumber; k++ )
                {
                    if ( ! channels[k] ) continue;

		    // if the pixel is not yet set or the comparator says that the next value is better than
		    // the one that has been set until now, set the new value as the resulting pixel value
                    if ( ! pixelSet || (*_comparator)( texture[ i * width + j ], channels[k][ i * width + j ] ) )
                    {
                        texture[ i * width + j ] = channels[k][ i * width + j ];
                        pixelSet = true;
                    }
                }
            }

        return texture;

    }

} /*namespace Viewer*/
} /*namespace M4D*/

#endif
