#ifndef INTENSITY_SUMMARIZER_SLICEVIEWER_TEXTURE_PREPARER_H
#error File IntensitySummarizerSliceViewerTexturePreparer.tcc cannot be included directly!
#else

namespace M4D
{
namespace Viewer
{

template< typename ElementType >
bool
IntensitySummarizerSliceViewerTexturePreparer< ElementType >
::prepare( const Imaging::InputPortList& inputPorts,
      uint32& width,
      uint32& height,
      GLint brightnessRate,
      GLint contrastRate,
      SliceOrientation so,
      uint32 slice,
      unsigned& dimension )
    {

	// get the dataset pixels
	ElementType** pixel = this->getDatasetArrays( inputPorts, SLICEVIEWER_INPUT_NUMBER, width, height, so, slice, dimension );

	bool datasetPresent = false;

	uint32 i;

	for ( i = 0; i < SLICEVIEWER_INPUT_NUMBER; i++ )
	    if ( pixel[i] )
	    {
		datasetPresent = true;

		// equalize intensities according to brightness- and contrastrate
		this->equalizeArray( pixel[i], width, height, brightnessRate, contrastRate );
	    }

	if ( ! datasetPresent )
	{
	    delete[] pixel;
	    return false;
	}

	// arrange their intensities
	ElementType* texture = IntensityArranger( pixel, SLICEVIEWER_INPUT_NUMBER, width, height );

	// equalize intensities of the resulting array according to brightness- and contrastrate
	this->equalizeArray( texture, width, height, brightnessRate, contrastRate );

	// prepare texture
        glTexImage2D( GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0,
                      GL_LUMINANCE, this->oglType(), texture );

	// free temporary allocated space
	delete[] texture;

	for ( i = 0; i < SLICEVIEWER_INPUT_NUMBER; i++ )
            if ( pixel[i] ) delete[] pixel[i];

	delete[] pixel;

        return true;
    }

} /*namespace Viewer*/
} /*namespace M4D*/

#endif
