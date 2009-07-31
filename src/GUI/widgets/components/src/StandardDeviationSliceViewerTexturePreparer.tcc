#ifndef STANDARD_DEVIATION_SLICEVIEWER_TEXTURE_PREPARER_H
#error File StandardDeviationSliceViewerTexturePreparer.tcc cannot be included directly!
#else

namespace M4D
{
namespace Viewer
{

template< typename ElementType >
ElementType*
StandardDeviationSliceViewerTexturePreparer< ElementType >
::IntensityArranger(
        ElementType** channels,
        uint32 channelNumber,
        uint32 width,
        uint32 height )
    {
        ElementType* texture = new ElementType[ width * height ];

        int32 pixelSet = -1;
	double currentSDValue = 0;
	double nextSDValue = 0;

	// loop through the input datasets for each position
        for ( uint32 i = 0; i < height; i++ )
            for ( uint32 j = 0; j < width; j++ )
            {
                pixelSet = -1;
		currentSDValue = nextSDValue = 0;
                for ( uint32 k = 0; k < channelNumber; k++ )
                {
                    if ( ! channels[k] ) continue;

		    nextSDValue = StandardDeviation( channels[k], j, i, width, height );

		    // if the pixel is not yet set or the SD value for the current position returns
		    // a greater number than for the previous largest, set the current position as the new candidate
                    if ( pixelSet == -1 || currentSDValue < nextSDValue )
                    {
                        currentSDValue = nextSDValue;
                        pixelSet = k;
                    }
                }

		texture[ i * width + j ] = channels[ pixelSet ][ i * width + j ];

            }

        return texture;

    }

template< typename ElementType >
double
StandardDeviationSliceViewerTexturePreparer< ElementType >
::StandardDeviation(
        ElementType* channel,
        uint32 xcoord,
        uint32 ycoord,
        uint32 width,
        uint32 height )
    {
	if ( (int32)xcoord < _range ||
	     (int32)ycoord < _range ||
	     xcoord >= width - _range ||
	     ycoord >= height - _range ) return 0;

	int32 i, j;

	double SDvalue = 0;
	double region = ( 2 * _range + 1 ) * ( 2 * _range + 1 );
	double center = channel[ ycoord * width + xcoord ];

	for ( i = -_range; i <= _range; ++i )
	    for ( j = -_range; j <= _range; ++j )
		SDvalue += ( (double)( channel[ ( ycoord + j ) * width + ( xcoord + i ) ] - center ) *
			   ( (double)channel[ ( ycoord + j ) * width + ( xcoord + i ) ] - center ) ) / region;

	return SDvalue;

    }

} /*namespace Viewer*/
} /*namespace M4D*/

#endif
