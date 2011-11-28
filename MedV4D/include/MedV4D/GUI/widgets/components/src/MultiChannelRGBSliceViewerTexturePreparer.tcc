#ifndef MULTI_CHANNEL_RGB_SLICEVIEWER_TEXTURE_PREPARER_H
#error File MultiChannelRGBSliceViewerTexturePreparer.tcc cannot be included directly!
#else

namespace M4D
{
namespace Viewer
{

template< typename ElementType >
bool
MultiChannelRGBSliceViewerTexturePreparer< ElementType >
::prepare( const Imaging::InputPortList& inputPorts,
      uint32& width,
      uint32& height,
      GLint brightnessRate,
      GLint contrastRate,
      SliceOrientation so,
      uint32 slice,
      unsigned& dimension )
    {

	ElementType** pixel = this->getDatasetArrays( inputPorts, SLICEVIEWER_INPUT_NUMBER, width, height, so, slice, dimension );

	uint32 i, j, k, textureCount = 0;

	// equalize arrays
	for ( i = 0; i < SLICEVIEWER_INPUT_NUMBER; i++ )
	    if ( pixel[i] )
	    {
		this->adjustArrayContrastBrightness( pixel[i], width, height, brightnessRate, contrastRate );
		textureCount++;
	    }

	if ( ! textureCount ) return false;

	ElementType* channelR = new ElementType[ width * height ];
	ElementType* channelG = new ElementType[ width * height ];
	ElementType* channelB = new ElementType[ width * height ];

	// loop through all the pixels
	for ( i = 0; i < height; i++ )
	    for ( j = 0; j < width; j++ )
	    {
		channelR[ i * width + j ] = channelG[ i * width + j ] = channelB[ i * width + j ] = 0;

		// give each channel's pixels a color, sum them up and divide them by the number of input datasets present
		for ( k = 0; k < SLICEVIEWER_INPUT_NUMBER; k++ )
		{
		    if ( ! pixel[k] ) continue;
		    if ( k < SLICEVIEWER_INPUT_NUMBER / 3 )
		    {
			channelR[ i * width + j ] += (ElementType)( ( ( 1.0 - (float32)( k % ( SLICEVIEWER_INPUT_NUMBER / 3 ) ) / (float32)( SLICEVIEWER_INPUT_NUMBER / 3 ) ) * pixel[k][ i * width + j ] ) / textureCount );
			channelG[ i * width + j ] += (ElementType)( ( ( (float32)( k % ( SLICEVIEWER_INPUT_NUMBER / 3 ) ) / (float32)( SLICEVIEWER_INPUT_NUMBER / 3 ) ) * pixel[k][ i * width + j ] ) / textureCount );
		    }
		    else if ( k < 2 * SLICEVIEWER_INPUT_NUMBER / 3 )
		    {
			channelG[ i * width + j ] += (ElementType)( ( ( 1.0 - (float32)( k % ( SLICEVIEWER_INPUT_NUMBER / 3 ) ) / (float32)( SLICEVIEWER_INPUT_NUMBER / 3 ) ) * pixel[k][ i * width + j ] ) / textureCount );
			channelB[ i * width + j ] += (ElementType)( ( ( (float32)( k % ( SLICEVIEWER_INPUT_NUMBER / 3 ) ) / (float32)( SLICEVIEWER_INPUT_NUMBER / 3 ) ) * pixel[k][ i * width + j ] ) / textureCount );
		    }
		    else
		    {
			channelB[ i * width + j ] += (ElementType)( ( ( 1.0 - (float32)( k % ( SLICEVIEWER_INPUT_NUMBER / 3 ) ) / (float32)( SLICEVIEWER_INPUT_NUMBER / 3 ) ) * pixel[k][ i * width + j ] ) / textureCount );
			channelR[ i * width + j ] += (ElementType)( ( ( (float32)( k % ( SLICEVIEWER_INPUT_NUMBER / 3 ) ) / (float32)( SLICEVIEWER_INPUT_NUMBER / 3 ) ) * pixel[k][ i * width + j ] ) / textureCount );
		    }
		}
	    }

	// arrange the channels of RGB
	ElementType* texture = RGBChannelArranger( channelR, channelG, channelB, width, height, brightnessRate, contrastRate, false );

	// prepare texture
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
                      GL_RGB, this->oglType(), texture );

	// free temporary allocated space
	delete[] texture;
	delete[] channelR;
	delete[] channelG;
	delete[] channelB;

	for ( i = 0; i < SLICEVIEWER_INPUT_NUMBER; i++ )
	    if ( pixel[i] ) delete[] pixel[i];

	delete[] pixel;

        return true;
    }

} /*namespace Viewer*/
} /*namespace M4D*/

#endif
