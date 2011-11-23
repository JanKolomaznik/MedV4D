#ifndef MAX_MIN_MIN_GRADIENT_RGB_SLICEVIEWER_TEXTURE_PREPARER_H
#error File MaxMinMinGradientRGBSliceViewerTexturePreparer.tcc cannot be included directly!
#else

namespace M4D
{
namespace Viewer
{

template< typename ElementType >
bool
MaxMinMinGradientRGBSliceViewerTexturePreparer< ElementType >
::prepare( const Imaging::InputPortList& inputPorts,
      uint32& width,
      uint32& height,
      GLint brightnessRate,
      GLint contrastRate,
      SliceOrientation so,
      uint32 slice,
      unsigned& dimension )
    {

	// get the original and the gradient values
	ElementType** pixel = SimpleSliceViewerTexturePreparer< ElementType >::getDatasetArrays( inputPorts, SLICEVIEWER_INPUT_NUMBER, width, height, so, slice, dimension );
        ElementType** gradient = GradientSliceViewerTexturePreparer< ElementType >::getDatasetArrays( inputPorts, SLICEVIEWER_INPUT_NUMBER, width, height, so, slice, dimension );

	uint32 i,textureCount = 0;

	for ( i = 0; i < SLICEVIEWER_INPUT_NUMBER; i++ )
	    if ( pixel[i] )
	    {
		textureCount++;
	    }

	if ( ! textureCount ) return false;

	// prepare the channels according to show the maximum, minimum gradient and minimum of the input dataset values
	ElementType* channelR = MaximumIntensitySliceViewerTexturePreparer< ElementType >::IntensityArranger( pixel, SLICEVIEWER_INPUT_NUMBER, width, height );
	ElementType* channelG = MinimumGradientSliceViewerTexturePreparer< ElementType >::IntensityArranger( gradient, SLICEVIEWER_INPUT_NUMBER, width, height );
	ElementType* channelB = MinimumIntensitySliceViewerTexturePreparer< ElementType >::IntensityArranger( pixel, SLICEVIEWER_INPUT_NUMBER, width, height );

	// set array values as RGB channel values
	ElementType* texture = RGBChannelArranger( channelR, channelG, channelB, width, height, brightnessRate, contrastRate );

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

	for ( i = 0; i < SLICEVIEWER_INPUT_NUMBER; i++ )
	    if ( gradient[i] ) delete[] gradient[i];

	delete[] pixel;
	delete[] gradient;

        return true;
    }

} /*namespace Viewer*/
} /*namespace M4D*/

#endif
