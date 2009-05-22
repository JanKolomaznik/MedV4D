#ifndef RGB_SLICEVIEWER_TEXTURE_PREPARER_H
#error File RGBSliceViewerTexturePreparer.tcc cannot be included directly!
#else

namespace M4D
{
namespace Viewer
{

template< typename ElementType >
ElementType*
RGBSliceViewerTexturePreparer< ElementType >
::RGBChannelArranger( 
	ElementType* channelR,
	ElementType* channelG,
	ElementType* channelB,
	uint32 width,
	uint32 height )
    {
	ElementType* texture = new ElementType[ width * height * 3 ];

	for ( uint32 i = 0; i < height; i++ )
	    for ( uint32 j = 0; j < width; j++ )
	    {
		if ( channelR ) texture[ i * width * 3 + j * 3 ] = channelR[ i * width + j ];
		else texture [ i * width * 3 + j * 3 ] = 0;

		if ( channelG ) texture[ i * width * 3 + j * 3 + 1 ] = channelG[ i * width + j ];
		else texture [ i * width * 3 + j * 3 + 1 ] = 0;

		if ( channelB ) texture[ i * width * 3 + j * 3 + 2 ] = channelB[ i * width + j ];
		else texture [ i * width * 3 + j * 3 + 2 ] = 0;
	    }

	return texture;

    }

template< typename ElementType >
bool
RGBSliceViewerTexturePreparer< ElementType >
::prepare( const Imaging::InputPortList& inputPorts,
      uint32& width,
      uint32& height,
      GLint brightnessRate,
      GLint contrastRate,
      SliceOrientation so,
      uint32 slice,
      unsigned& dimension )
    {

	ElementType** pixel = this->getDatasetArrays( inputPorts, 3, width, height, brightnessRate, contrastRate, so, slice, dimension );

	if ( ! pixel[0] && ! pixel[1] && ! pixel[2] )
	{
	    delete[] pixel;
	    return false;
	}

	ElementType* texture = RGBChannelArranger( pixel[0], pixel[1], pixel[2], width, height );

        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
                      GL_RGB, this->oglType(), texture );

	delete[] texture;
	if ( pixel[0] ) delete[] pixel[0];
	if ( pixel[1] ) delete[] pixel[1];
	if ( pixel[2] ) delete[] pixel[2];

	delete[] pixel;

        return true;
    }

} /*namespace Viewer*/
} /*namespace M4D*/

#endif
