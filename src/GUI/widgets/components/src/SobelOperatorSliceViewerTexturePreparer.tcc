#ifndef SOBEL_OPERATOR_SLICEVIEWER_TEXTURE_PREPARER_H
#error File SobelOperatorSliceViewerTexturePreparer.tcc cannot be included directly!
#else

namespace M4D
{
namespace Viewer
{

template< typename ElementType >
ElementType*
SobelOperatorSliceViewerTexturePreparer< ElementType >
::IntensityArranger(
        ElementType** channels,
        uint32 channelNumber,
        uint32 width,
        uint32 height )
    {
        ElementType* texture = new ElementType[ width * height ];

        int32 pixelSet = -1;
	ElementType currentSobelValue = 0;
	ElementType nextSobelValue = 0;

	// loop through the input datasets for each position
        for ( uint32 i = 0; i < height; i++ )
            for ( uint32 j = 0; j < width; j++ )
            {
                pixelSet = -1;
		currentSobelValue = nextSobelValue = 0;
                for ( uint32 k = 0; k < channelNumber; k++ )
                {
                    if ( ! channels[k] ) continue;

		    nextSobelValue = SobelOperator( channels[k], j, i, width, height );

		    // if the pixel is not yet set or the Sobel Operator for the current position returns
		    // a greater number than for the previous largest, set the current position as the new candidate
                    if ( pixelSet == -1 || currentSobelValue < nextSobelValue )
                    {
                        currentSobelValue = nextSobelValue;
                        pixelSet = k;
                    }
                }

		texture[ i * width + j ] = channels[ pixelSet ][ i * width + j ];

            }

        return texture;

    }

template< typename ElementType >
ElementType
SobelOperatorSliceViewerTexturePreparer< ElementType >
::SobelOperator(
        ElementType* channel,
        uint32 xcoord,
        uint32 ycoord,
        uint32 width,
        uint32 height )
    {
	if ( xcoord <= 0 ||
	     ycoord <= 0 ||
	     xcoord >= width - 1 ||
	     ycoord >= height - 1 ) return 0;

	ElementType Gy = channel[ ( ycoord + 1 ) * width + ( xcoord + 1 ) ] +
			 2 * channel[ ( ycoord + 1 ) * width + xcoord ] +
			 channel[ ( ycoord + 1 ) * width + ( xcoord - 1 ) ] -
			 channel[ ( ycoord - 1 ) * width + ( xcoord + 1 ) ] -
			 2 * channel[ ( ycoord + 1 ) * width + xcoord ] -
			 channel[ ( ycoord + 1 ) * width + ( xcoord - 1 ) ];

	ElementType Gx = channel[ ( ycoord + 1 ) * width + ( xcoord + 1 ) ] +
			 2 * channel[ ycoord * width + ( xcoord + 1 ) ] +
			 channel[ ( ycoord - 1 ) * width + ( xcoord + 1 ) ] -
			 channel[ ( ycoord + 1 ) * width + ( xcoord - 1 ) ] -
			 2 * channel[ ycoord  * width + ( xcoord - 1 ) ] -
			 channel[ ( ycoord - 1 ) * width + ( xcoord - 1 ) ];

	return std::sqrt( Gx * Gx + Gy * Gy );

    }

} /*namespace Viewer*/
} /*namespace M4D*/

#endif
