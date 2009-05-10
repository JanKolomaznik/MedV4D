#ifndef SIMPLE_SLICEVIEWER_TEXTURE_PREPARER_H
#error File SimpleSliceViewerTexturePreparer.tcc cannot be included directly!
#else

namespace M4D
{
namespace Viewer
{

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
        uint32 depth;
        double maxvalue;
        bool unsgn;
        Imaging::InputPortTyped<Imaging::AbstractImage>* inPort = inputPorts.GetPortTypedSafe< Imaging::InputPortTyped<Imaging::AbstractImage> >( 0 );

        // get the maximum value of the given element type
        if ( typeid( ElementType ) == typeid( uint8 ) || typeid( ElementType ) == typeid( uint16 ) || typeid( ElementType ) == typeid( uint32 ) || typeid( ElementType ) == typeid( uint64 ) )
        {
            maxvalue = std::pow( (double)256, (double)sizeof( ElementType ) ) - 1;
            unsgn = true;
        }
        else
        {
            maxvalue = (int)( std::pow( (double)256, (double)sizeof( ElementType ) ) / 2 - 1 );
            unsgn = false;
        }

        int32 xstride, ystride, zstride;
        bool ready = true;
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
                if ( !ready ) return ready;
            }
            else
            {
                ready = false;
                return ready;
            }
        }
        catch (...) { ready = false; }
        if ( !ready ) return ready;

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
                    if ( DISPLAY_PIXEL_VALUE( pixel[ i * newWidth + j ], mean, brightnessRate, 1 ) > maxvalue ) pixel[ i * newWidth + j ] = (ElementType)maxvalue;
                    else if ( DISPLAY_PIXEL_VALUE( pixel[ i * newWidth + j ], mean, brightnessRate, 1 ) < ( unsgn ? 0 : -maxvalue ) ) pixel[ i * newWidth + j ] = ( unsgn ? 0 : (ElementType)(-maxvalue) );
                    else pixel[ i * newWidth + j ] = (ElementType)( DISPLAY_PIXEL_VALUE( pixel[ i * newWidth + j ], mean, brightnessRate, 1 ) );
                }
                // if extra pixels are reached
                else
                    pixel[ i * newWidth + j ] = ( unsgn ? 0 : (ElementType)(-maxvalue) );
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
                if ( DISPLAY_PIXEL_VALUE( pixel[ i * newWidth + j ], mean, 0, cont ) > maxvalue ) pixel[ i * newWidth + j ] = (ElementType)maxvalue;
                else if ( DISPLAY_PIXEL_VALUE( pixel[ i * newWidth + j ], mean, 0, cont ) < ( unsgn ? 0 : -maxvalue ) ) pixel[ i * newWidth + j ] = ( unsgn ? 0 : (ElementType)(-maxvalue) );
                else pixel[ i * newWidth + j ] = (ElementType)( DISPLAY_PIXEL_VALUE( pixel[ i * newWidth + j ], mean, 0, cont ) );
            }

        width = newWidth;
        height = newHeight;

        glTexImage2D( GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0,
                      GL_LUMINANCE, this->oglType(), pixel );
        delete[] pixel;
        return ready;
    }

} /*namespace Viewer*/
} /*namespace M4D*/

#endif
