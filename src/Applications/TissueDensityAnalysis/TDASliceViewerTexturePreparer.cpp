/**
 *  @ingroup TDA
 *	@author Milan Lepik
 *  @file TDASliceViewerTexturePreparer.cpp
 */

#include "TDASliceViewerTexturePreparer.h"

using namespace M4D;
using namespace M4D::Viewer;

template< typename ElementType >
void
TDASliceViewerTexturePreparer< ElementType >
::copy( ElementType* dst, ElementType* src, uint32 width, uint32 height, uint32 newWidth, uint32 newHeight, uint32 depth, int32 xstride, int32 ystride, int32 zstride )
{
    uint32 i, j;
    for ( i = 0; i < newHeight; i++ )
        for ( j = 0; j < newWidth; j++ )
	if ( i < height && j < width ) dst[ i * newWidth + j ] = src[ j * xstride + i * ystride + depth * zstride ];
	else dst[ i * newWidth + j ] = 0;
}

template< typename ElementType >
void 
TDASliceViewerTexturePreparer< ElementType >
::maskCopy( ElementType* dst, ElementType* src, ElementType* mask, uint32 width, uint32 height, uint32 newWidth, uint32 newHeight, uint32 depth, int32 xstride, int32 ystride, int32 zstride )
{
    uint32 i, j;
    for ( i = 0; i < newHeight; i++ )
        for ( j = 0; j < newWidth; j++ )
			if ( i < height && j < width && mask[ j * xstride + i * ystride + depth * zstride ] > 0) 
				dst[ i * newWidth + j ] = src[ j * xstride + i * ystride + depth * zstride ];
			else 
				dst[ i * newWidth + j ] = 0;
}



template< typename ElementType >
bool
TDASliceViewerTexturePreparer< ElementType >
::prepare( const Imaging::InputPortList& inputPorts,
      uint32& width,
      uint32& height,
      GLint brightnessRate,
      GLint contrastRate,
      SliceOrientation so,
      uint32 slice,
      unsigned& dimension )
    {

	// get the input datasets
      ElementType** pixel = getDatasetArrays( inputPorts, 1, width, height, so, slice, dimension );

	if ( ! *pixel )
	{
	    delete[] pixel;
	    return false;
	}

	// equalize the first input array
	adjustArrayContrastBrightness( *pixel, width, height, brightnessRate, contrastRate );

	// prepare texture
        glTexImage2D( GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0,
                      GL_LUMINANCE, this->oglType(), *pixel );

	// free temporary allocated space
	delete[] *pixel;

	delete[] pixel;

        return true;
    }

template< typename ElementType >
ElementType**
TDASliceViewerTexturePreparer< ElementType >
::getDatasetArrays( const Imaging::InputPortList& inputPorts,
      uint32 numberOfDatasets,
      uint32& width,
      uint32& height,
      SliceOrientation so,
      uint32 slice,
      unsigned& dimension )
{
	uint32 i, tmpwidth, tmpheight;
	Imaging::InputPortTyped<Imaging::AImage>* inPort;
	Imaging::InputPortTyped<Imaging::AImage>* inMaskPort;
	ElementType** result = new ElementType*[ numberOfDatasets ];

	width = height = 0;

	// loop through the input ports
	for ( i = 0; i < numberOfDatasets; i++ )
	{
	    if ( inputPorts.Size() <= i ) result[i] = NULL;
	    else
	    {
			tmpwidth = tmpheight = 0;

			// get the port and drag the data out of the port
			inPort = inputPorts.GetPortTypedSafe< Imaging::InputPortTyped<Imaging::AImage> >( i );
			inMaskPort = inputPorts.GetPortTypedSafe< Imaging::InputPortTyped<Imaging::AImage> >( i + 1 );
			result[i] = prepareSingle( inPort, inMaskPort, tmpwidth, tmpheight, so, slice, dimension );
            	if ( result[i] && ( ( tmpwidth < width && tmpwidth > 0 ) || width == 0 ) ) width = tmpwidth;
            	if ( result[i] && ( ( tmpheight < height && tmpheight > 0 ) || height == 0 ) ) height = tmpheight;
	    }
	}

	return result;
}


template< typename ElementType >
ElementType*
TDASliceViewerTexturePreparer< ElementType >
::prepareSingle(
		Imaging::InputPortTyped<Imaging::AImage>* inPort,
		Imaging::InputPortTyped<Imaging::AImage>* inMaskPort,
		uint32& width,
		uint32& height,
		SliceOrientation so,
		uint32 slice,
		unsigned& dimension )
    {
        bool ready = true;
        int32 xstride = 0, ystride = 0, zstride = 0;
        uint32 depth = 0;
        ElementType* pixel = 0, *original = 0, *mask = 0;
        try
        {
            // need to lock dataset first
            if ( inPort->TryLockDataset() )
            {
                try
                {
                    // check dimension
                    if ( inPort->GetDatasetTyped().GetDimension() == 2 )
                    {
                        Vector< uint32, 2 > size;
                        Vector< int32, 2 > strides;
                        original = Imaging::Image< ElementType, 2 >::Cast(inPort->GetDatasetTyped()).GetPointer( size, strides );
						mask = Imaging::Image< ElementType, 2 >::Cast(inMaskPort->GetDatasetTyped()).GetPointer( size, strides );
                        width = size[0];
                        height = size[1];
                        xstride = strides[0];
                        ystride = strides[1];
                        dimension = 2;
                        depth = zstride = 0;
                        slice = 0;
                    }
                    else if ( inPort->GetDatasetTyped().GetDimension() == 3 )
                    {
                        dimension = 3;
                        Vector< uint32, 3 > size;
                        Vector< int32, 3 > strides;

                        // check orientation
                        switch ( so )
                        {
                            case xy:
                            {
                  //            int it=inPort->GetDatasetTyped().GetElementTypeID();
                  //            int imt=inMaskPort->GetDatasetTyped().GetElementTypeID();

                                original = Imaging::Image< ElementType, 3 >::Cast(inPort->GetDatasetTyped()).GetPointer( size, strides );
								mask = Imaging::Image< ElementType, 3 >::Cast(inMaskPort->GetDatasetTyped()).GetPointer( size, strides );
                                width = size[0];
                                height = size[1];
                                depth = size[2];
                                xstride = strides[0];
                                ystride = strides[1];
                                zstride = strides[2];
                                break;
                            }

                            case yz:
                            {
                                original = Imaging::Image< ElementType, 3 >::Cast(inPort->GetDatasetTyped()).GetPointer( size, strides );
								mask = Imaging::Image< ElementType, 3 >::Cast(inMaskPort->GetDatasetTyped()).GetPointer( size, strides );
                                width = size[1];
                                height = size[2];
                                depth = size[0];
                                xstride = strides[1];
                                ystride = strides[2];
                                zstride = strides[0];
                                break;
                            }

                            case zx:
                            {
                                original = Imaging::Image< ElementType, 3 >::Cast(inPort->GetDatasetTyped()).GetPointer( size, strides );
								mask = Imaging::Image< ElementType, 3 >::Cast(inMaskPort->GetDatasetTyped()).GetPointer( size, strides );
                                width = size[2];
                                height = size[0];
                                depth = size[1];
                                xstride = strides[2];
                                ystride = strides[0];
                                zstride = strides[1];
                                break;
                            }
                        }
                    }
                    else
                    {
                        ready = false;
                        original = 0;
                    }

                    if ( !original ) ready = false;

        	    else
		    {

			// check to see if modification is required for power of 2 long and wide texture
		        float power_of_two_width_ratio=std::log((float)(width))/std::log(2.0);
        		float power_of_two_height_ratio=std::log((float)(height))/std::log(2.0);

       			uint32 newWidth=(uint32)std::pow( (double)2.0, (double)std::ceil(power_of_two_width_ratio) );
        		uint32 newHeight=(uint32)std::pow( (double)2.0, (double)std::ceil(power_of_two_height_ratio) );

        		pixel = new ElementType[ newHeight * newWidth ];

        		maskCopy( pixel, original, mask, width, height, newWidth, newHeight, slice, xstride, ystride, zstride );

        		width = newWidth;
        		height = newHeight;

		    }

                } catch (...) { ready = false; }
                inPort->ReleaseDatasetLock();
                if ( !ready ) return NULL;
            }
            else
            {
                ready = false;
                return NULL;
            }
        }
        catch (...) { ready = false; }
        if ( !ready ) return NULL;

	return pixel;
}


