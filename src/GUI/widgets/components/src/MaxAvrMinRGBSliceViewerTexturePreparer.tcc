#ifndef MAX_AVR_MIN_RGB_SLICEVIEWER_TEXTURE_PREPARER_H
#error File MaxAvrMinRGBSliceViewerTexturePreparer.tcc cannot be included directly!
#else

namespace M4D
{
namespace Viewer
{

template< typename ElementType >
bool
MaxAvrMinRGBSliceViewerTexturePreparer< ElementType >
::prepare( const Imaging::InputPortList& inputPorts,
      uint32& width,
      uint32& height,
      GLint brightnessRate,
      GLint contrastRate,
      SliceOrientation so,
      uint32 slice,
      unsigned& dimension )
    {

	// get the original values
	ElementType** pixel = this->getDatasetArrays( inputPorts, SLICEVIEWER_INPUT_NUMBER, width, height, so, slice, dimension );

	uint32 i,textureCount = 0;

	for ( i = 0; i < SLICEVIEWER_INPUT_NUMBER; i++ )
	    if ( pixel[i] )
	    {
		textureCount++;
	    }

	if ( ! textureCount ) return false;

	// prepare the channels according to show the maximum, average and minimum of the input dataset values
	ElementType* channelR = MaximumIntensitySliceViewerTexturePreparer< ElementType >::IntensityArranger( pixel, SLICEVIEWER_INPUT_NUMBER, width, height );
	ElementType* channelG = AverageIntensitySliceViewerTexturePreparer< ElementType >::IntensityArranger( pixel, SLICEVIEWER_INPUT_NUMBER, width, height );
	ElementType* channelB = MinimumIntensitySliceViewerTexturePreparer< ElementType >::IntensityArranger( pixel, SLICEVIEWER_INPUT_NUMBER, width, height );

	// equalize arrays
	this->equalizeArray( channelR, width, height, brightnessRate, contrastRate );
        this->equalizeArray( channelG, width, height, brightnessRate, contrastRate );
        this->equalizeArray( channelB, width, height, brightnessRate, contrastRate );

	// arrange the intensities as RGB channels
	ElementType* texture = RGBChannelArranger( channelR, channelG, channelB, width, height );

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
