#ifndef RGB_SLICEVIEWER_TEXTURE_PREPARER_H
#error File RGBSliceViewerTexturePreparer.tcc cannot be included directly!
#else

namespace M4D
{
namespace Viewer
{

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
        Imaging::InputPortTyped<Imaging::AbstractImage>* inPort = inputPorts.GetPortTypedSafe< Imaging::InputPortTyped<Imaging::AbstractImage> >( 0 );

	ElementType* pixel[3];
	pixel[0] = this->prepareSingle( inPort, width, height, brightnessRate, contrastRate, so, slice, dimension );
	uint32 tmpwidth, tmpheight;

	if ( inputPorts.Size() > 1 )
	{
	    inPort = inputPorts.GetPortTypedSafe< Imaging::InputPortTyped<Imaging::AbstractImage> >( 1 );
	    pixel[1] = this->prepareSingle( inPort, tmpwidth, tmpheight, brightnessRate, contrastRate, so, slice, dimension );
	    if ( ! pixel[0] || ( pixel[1] && tmpwidth < width ) ) width = tmpwidth;
	    if ( ! pixel[0] || ( pixel[1] && tmpheight < height ) ) height = tmpheight;
	}
	else pixel[1] = 0;

	if ( inputPorts.Size() > 2 )
	{
	    inPort = inputPorts.GetPortTypedSafe< Imaging::InputPortTyped<Imaging::AbstractImage> >( 2 );
	    pixel[2] = this->prepareSingle( inPort, tmpwidth, tmpheight, brightnessRate, contrastRate, so, slice, dimension );
	    if ( ( ! pixel[0] && ! pixel[1] ) || ( pixel[2] && tmpwidth < width ) ) width = tmpwidth;
	    if ( ( ! pixel[0] && ! pixel[1] ) || ( pixel[2] && tmpheight < height ) ) height = tmpheight;
	}
	else pixel[2] = 0;

	if ( ! pixel[0] && ! pixel[1] && ! pixel[2] ) return false;

	ElementType* texture = new ElementType[ width * height * 3 ];

	for ( uint32 i = 0; i < height; i++ )
	    for ( uint32 j = 0; j < width; j++ )
	    {
		if ( pixel[0] ) texture[ i * width * 3 + j * 3 ] = pixel[0][ i * width + j ];
		else texture [ i * width * 3 + j * 3 ] = 0;

		if ( pixel[1] ) texture[ i * width * 3 + j * 3 + 1 ] = pixel[1][ i * width + j ];
		else texture [ i * width * 3 + j * 3 + 1 ] = 0;

		if ( pixel[2] ) texture[ i * width * 3 + j * 3 + 2 ] = pixel[2][ i * width + j ];
		else texture [ i * width * 3 + j * 3 + 2 ] = 0;
	    }

        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
                      GL_RGB, this->oglType(), texture );
        return true;
    }

} /*namespace Viewer*/
} /*namespace M4D*/

#endif
