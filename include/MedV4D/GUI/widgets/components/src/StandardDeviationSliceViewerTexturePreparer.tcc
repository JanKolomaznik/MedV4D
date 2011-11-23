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
	if ( (int32)xcoord < _radius ||
	     (int32)ycoord < _radius ||
	     xcoord >= width - _radius ||
	     ycoord >= height - _radius ) return 0;

	int32 i, j;

	double SDvalue = 0;
	double expectedValue = 0;
	double region = ( 2 * _radius + 1 ) * ( 2 * _radius + 1 );

	for ( i = -_radius; i <= _radius; ++i )
	    for ( j = -_radius; j <= _radius; ++j )
		expectedValue += ( (double)channel[ ( ycoord + j ) * width + ( xcoord + i ) ] ) / region;

	for ( i = -_radius; i <= _radius; ++i )
	    for ( j = -_radius; j <= _radius; ++j )
		SDvalue += ( (double)( channel[ ( ycoord + j ) * width + ( xcoord + i ) ] - expectedValue ) *
			   ( (double)channel[ ( ycoord + j ) * width + ( xcoord + i ) ] - expectedValue ) ) / region;

	return SDvalue;

    }

} /*namespace Viewer*/
} /*namespace M4D*/

#endif
