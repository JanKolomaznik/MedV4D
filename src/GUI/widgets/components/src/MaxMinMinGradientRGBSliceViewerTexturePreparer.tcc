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

	ElementType** pixel = SimpleSliceViewerTexturePreparer< ElementType >::getDatasetArrays( inputPorts, SLICEVIEWER_INPUT_NUMBER, width, height, so, slice, dimension );
        ElementType** gradient = GradientSliceViewerTexturePreparer< ElementType >::getDatasetArrays( inputPorts, SLICEVIEWER_INPUT_NUMBER, width, height, so, slice, dimension );

	uint32 i,textureCount = 0;

	for ( i = 0; i < SLICEVIEWER_INPUT_NUMBER; i++ )
	    if ( pixel[i] )
	    {
		textureCount++;
	    }

	if ( ! textureCount ) return false;

	ElementType* channelR = MaximumIntensitySliceViewerTexturePreparer< ElementType >::IntensityArranger( pixel, SLICEVIEWER_INPUT_NUMBER, width, height );
	ElementType* channelG = MinimumGradientSliceViewerTexturePreparer< ElementType >::IntensityArranger( gradient, SLICEVIEWER_INPUT_NUMBER, width, height );
	ElementType* channelB = MinimumIntensitySliceViewerTexturePreparer< ElementType >::IntensityArranger( pixel, SLICEVIEWER_INPUT_NUMBER, width, height );
	
	this->equalizeArray( channelR, width, height, brightnessRate, contrastRate );
	this->equalizeArray( channelG, width, height, brightnessRate, contrastRate );
	this->equalizeArray( channelB, width, height, brightnessRate, contrastRate );

	ElementType* texture = RGBChannelArranger( channelR, channelG, channelB, width, height );

        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
                      GL_RGB, this->oglType(), texture );

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
