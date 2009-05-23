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
::IntensitySelectorArranger(
        ElementType** channels,
        uint32 channelNumber,
        uint32 width,
        uint32 height )
    {
	ElementType* texture = new ElementType[ width * height ];

	bool pixelSet = false;

	for ( uint32 i = 0; i < height; i++ )
	    for ( uint32 j = 0; j < width; j++ )
	    {
		pixelSet = false;
		texture[ i * width + j ] = TypeTraits< ElementType >::Min;
		for ( uint32 k = 0; k < channelNumber; k++ )
		{
		    if ( ! channels[k] ) continue;
		    if ( ! pixelSet || (*_comparator)( texture[ i * width + j ], channels[k][ i * width + j ] ) )
		    {
			texture[ i * width + j ] = channels[k][ i * width + j ];
			pixelSet = true;
		    }
		}
	    }

	return texture;

    }

template< typename ElementType >
bool
IntensitySelectorSliceViewerTexturePreparer< ElementType >
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

	bool datasetPresent = false;

	uint32 i;

	for ( i = 0; i < SLICEVIEWER_INPUT_NUMBER; i++ )
	    if ( pixel[i] )
	    {
		datasetPresent = true;
		break;
	    }

	if ( ! datasetPresent )
	{
	    delete[] pixel;
	    return false;
	}

	ElementType* texture = IntensitySelectorArranger( pixel, SLICEVIEWER_INPUT_NUMBER, width, height );

	this->equalizeArray( texture, width, height, brightnessRate, contrastRate );

        glTexImage2D( GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0,
                      GL_LUMINANCE, this->oglType(), texture );

	delete[] texture;

	for ( i = 0; i < SLICEVIEWER_INPUT_NUMBER; i++ )
            if ( pixel[i] ) delete[] pixel[i];

	delete[] pixel;

        return true;
    }

} /*namespace Viewer*/
} /*namespace M4D*/

#endif
