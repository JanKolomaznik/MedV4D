#ifndef GRADIENT_SLICEVIEWER_TEXTURE_PREPARER_H
#error File GradientSliceViewerTexturePreparer.tcc cannot be included directly!
#else

namespace M4D
{
namespace Viewer
{

template< typename ElementType >
ElementType**
GradientSliceViewerTexturePreparer< ElementType >
::getDatasetArrays( const Imaging::InputPortList& inputPorts,
      uint32 numberOfDatasets,
      uint32& width,
      uint32& height,
      SliceOrientation so,
      uint32 slice,
      unsigned& dimension )
    {
	ElementType** image = SimpleSliceViewerTexturePreparer< ElementType >::getDatasetArrays( inputPorts,
										   numberOfDatasets,
										   width,
										   height,
										   so,
										   slice,
										   dimension );
	ElementType** result = new ElementType*[ numberOfDatasets ];

	int32 i, k, l;
	ElementType* tmp = 0;

	for ( i = (int32)numberOfDatasets - 1; i >= 0; i-- )
	{
	    if ( image[i] == 0 )
	    {
	        result[i] = 0;
	    }
	    else if ( image[i] != 0 && tmp == 0 )
	    {
	        result[i] = 0;
	        tmp = image[i];
	    }
	    else
	    {
	        result[i] = new ElementType[ width * height ];
	        for ( k = 0; k < (int32)height; ++k )
	            for ( l = 0; l < (int32)width; ++l )
	            {
			if ( TypeTraits< ElementType >::Min == 0 ) result[ i ][ k * width + l ] = std::abs( tmp[ k * width + l ] - image[ i ][ k * width + l ] );
			else result[ i ][ k * width + l ] = tmp[ k * width + l ] - image[ i ][ k * width + l ];
	    	    }

	        tmp = image[i];
	    }
	}


	for ( i = 0; i < (int32)numberOfDatasets; i++ )
	    if ( image[i] ) delete image[i];

	delete[] image;


        return result;

    }

} /*namespace Viewer*/
} /*namespace M4D*/

#endif
