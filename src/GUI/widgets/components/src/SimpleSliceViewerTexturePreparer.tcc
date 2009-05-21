#ifndef SIMPLE_SLICEVIEWER_TEXTURE_PREPARER_H
#error File SimpleSliceViewerTexturePreparer.tcc cannot be included directly!
#else

namespace M4D
{
namespace Viewer
{

template< typename ElementType >
ElementType*
SimpleSliceViewerTexturePreparer< ElementType >
::prepareSingle( Imaging::InputPortTyped<Imaging::AbstractImage>* inPort,
      uint32& width,
      uint32& height,
      GLint brightnessRate,
      GLint contrastRate,
      SliceOrientation so,
      uint32 slice,
      unsigned& dimension )
    {
        bool ready = true;
        int32 xstride, ystride, zstride;
        uint32 depth;
        ElementType* pixel, *original;
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
                        original = Imaging::Image< ElementType, 2 >::CastAbstractImage(inPort->GetDatasetTyped()).GetPointer( size, strides );
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
                                original = Imaging::Image< ElementType, 3 >::CastAbstractImage(inPort->GetDatasetTyped()).GetPointer( size, strides );
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
                                original = Imaging::Image< ElementType, 3 >::CastAbstractImage(inPort->GetDatasetTyped()).GetPointer( size, strides );
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
                                original = Imaging::Image< ElementType, 3 >::CastAbstractImage(inPort->GetDatasetTyped()).GetPointer( size, strides );
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

        // check to see if modification is required for power of 2 long and wide texture
        float power_of_two_width_ratio=std::log((float)(width))/std::log(2.0);
        float power_of_two_height_ratio=std::log((float)(height))/std::log(2.0);

        uint32 newWidth=(uint32)std::pow( (double)2.0, (double)std::ceil(power_of_two_width_ratio) );
        uint32 newHeight=(uint32)std::pow( (double)2.0, (double)std::ceil(power_of_two_height_ratio) );

        pixel = new ElementType[ newHeight * newWidth ];

        copy( pixel, original, width, height, newWidth, slice, xstride, ystride, zstride );
        uint32 i, j;
        double mean, cont;
        mean = 0.;
        for ( i = 0; i < height; i++ )
            for ( j = 0; j < width; j++ ) mean += (double)pixel[ i * newWidth + j ] / (double)(width*height);
        brightnessRate -= (int)mean;
        for ( i = 0; i < newHeight; ++i )
            for ( j = 0; j < newWidth; j++ )
            {
                // if inside the image
                if ( i < height && j < width )
                {
                    if ( DISPLAY_PIXEL_VALUE( pixel[ i * newWidth + j ], mean, brightnessRate, 1 ) > TypeTraits< ElementType >::Max ) pixel[ i * newWidth + j ] = TypeTraits< ElementType >::Max;
                    else if ( DISPLAY_PIXEL_VALUE( pixel[ i * newWidth + j ], mean, brightnessRate, 1 ) < TypeTraits< ElementType >::Min ) pixel[ i * newWidth + j ] = TypeTraits< ElementType >::Min;
                    else pixel[ i * newWidth + j ] = (ElementType)( DISPLAY_PIXEL_VALUE( pixel[ i * newWidth + j ], mean, brightnessRate, 1 ) );
                }
                // if extra pixels are reached
                else
                    pixel[ i * newWidth + j ] = TypeTraits< ElementType >::Min;
            }

        mean = 0.;
        for ( i = 0; i < height; i++ )
            for ( j = 0; j < width; j++ ) mean += (double)pixel[ i * newWidth + j ] / (double)(width*height);

        cont = 0.;
        for ( i = 0; i < height; i++ )
            for ( j = 0; j < width; j++ ) cont += std::abs( (double)( pixel[ i * newWidth + j ] - mean ) ) / (double)(width*height);

        if ( cont != 0 ) cont = (double)contrastRate/cont;
        for ( i = 0; i < height; i++ )
            for ( j = 0; j < width; j++ )
            {
                if ( DISPLAY_PIXEL_VALUE( pixel[ i * newWidth + j ], mean, 0, cont ) > TypeTraits< ElementType >::Max ) pixel[ i * newWidth + j ] = TypeTraits< ElementType >::Max;
                else if ( DISPLAY_PIXEL_VALUE( pixel[ i * newWidth + j ], mean, 0, cont ) < TypeTraits< ElementType >::Min ) pixel[ i * newWidth + j ] = TypeTraits< ElementType >::Min;
                else pixel[ i * newWidth + j ] = (ElementType)( DISPLAY_PIXEL_VALUE( pixel[ i * newWidth + j ], mean, 0, cont ) );
            }

        width = newWidth;
        height = newHeight;

	return pixel;
    }

template< typename ElementType >
bool
SimpleSliceViewerTexturePreparer< ElementType >
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

	ElementType* pixel = prepareSingle( inPort, width, height, brightnessRate, contrastRate, so, slice, dimension );

	if ( ! pixel ) return false;

        glTexImage2D( GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0,
                      GL_LUMINANCE, this->oglType(), pixel );
        return true;
    }

} /*namespace Viewer*/
} /*namespace M4D*/

#endif
