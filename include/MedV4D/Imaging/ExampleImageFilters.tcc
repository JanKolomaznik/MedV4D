/**
 * @ingroup imaging
 * @author Jan Kolomaznik
 * @file ExampleImageFilters.tcc
 * @{
 **/

#ifndef _EXAMPLE_IMAGE_FILTERS_H
#error File ExampleImageFilters.tcc cannot be included directly!
#else

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging {

template< typename InputElementType, typename OutputElementType >
CopyImageFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
::CopyImageFilter() : PredecessorType ( 0, 10 )
{

}

template< typename InputElementType, typename OutputElementType >
bool
CopyImageFilter< Image< InputElementType, 3 >, Image< OutputElementType, 3 > >
::ProcessSlice (
        const Image< InputElementType, 3 > 	&in,
        Image< OutputElementType, 3 >		&out,
        size_t			x1,
        size_t			y1,
        size_t			x2,
        size_t			y2,
        size_t			slice
)
{
        if ( !this->CanContinue() ) { //Someone wants filter to stop.
                return false;
        }
        for ( size_t i = x1; i < x2; ++i ) {
                for ( size_t j = y1; j < y2; ++j ) {
                        out.GetElement ( i, j, slice ) = in.GetElement ( i, j, slice );
                }
        }
        return true;
}

//*****************************************************************************
//*****************************************************************************

template< typename ElementType >
ColumnMaxImageFilter< ElementType >
::ColumnMaxImageFilter()
{

}

template< typename ElementType >
bool
ColumnMaxImageFilter< ElementType >
::ProcessVolume (
        const Image< ElementType, 3 > 		&in,
        Image< ElementType, 2 >			&out,
        size_t					x1,
        size_t					y1,
        size_t					z1,
        size_t					x2,
        size_t					y2,
        size_t					z2
)
{
        for ( size_t i = x1; i < x2; ++i ) {
                if ( !this->CanContinue() ) { //Someone wants filter to stop.
                        return false;
                }
                for ( size_t j = y1; j < y2; ++j ) {
                        ElementType max = in.GetElement ( i, j, z1 );
                        for ( size_t k = z1+1; k < z2; ++k ) {
                                if ( in.GetElement ( i, j, k ) > max ) {
                                        max = in.GetElement ( i, j, k );
                                }
                        }
                        out.GetElement ( i, j ) = max;
                }
        }

        return true;
}

template< typename ElementType >
void
ColumnMaxImageFilter< ElementType >
::PrepareOutputDatasets()
{
        PredecessorType::PrepareOutputDatasets();

        int32 minimums[2];
        int32 maximums[2];
        float32 pixelExtents[2];

        for ( unsigned i=0; i < 2; ++i ) {
                const DimensionExtents & dimExt = this->in->GetDimensionExtents ( i );

                minimums[i] = dimExt.minimum;
                maximums[i] = dimExt.maximum;
                pixelExtents[i] = dimExt.elementExtent;
        }
        this->SetOutputImageSize ( minimums, maximums, pixelExtents );
}


//*****************************************************************************
//*****************************************************************************

template< typename InputElementType >
SimpleThresholdingImageFilter< Image< InputElementType, 3 > >
::SimpleThresholdingImageFilter()
{
        _intervalTop = 0;
        _intervalBottom = 0;
        _defaultValue = 0;
}

template< typename InputElementType >
bool
SimpleThresholdingImageFilter< Image< InputElementType, 3 > >
::ProcessSlice (
        const Image< InputElementType, 3 > 	&in,
        Image< InputElementType, 3 >		&out,
        size_t			x1,
        size_t			y1,
        size_t			x2,
        size_t			y2,
        size_t			slice
)
{
        if ( !this->CanContinue() ) { //Someone wants filter to stop.
                return false;
        }
        for ( size_t i = x1; i < x2; ++i ) {
                for ( size_t j = y1; j < y2; ++j ) {
                        InputElementType val = in.GetElement ( i, j, slice );
                        if ( val > _intervalTop || val < _intervalBottom )
                                val = _defaultValue;
                        out.GetElement ( i, j, slice ) = val;
                }
        }
        return true;
}

//*****************************************************************************
//*****************************************************************************

template< typename InputElementType >
SimpleConvolutionImageFilter< Image< InputElementType, 3 > >
::SimpleConvolutionImageFilter() : PredecessorType ( 0, 5 ) //TODO - constructor
{
        _side = 5;
        _matrix = new float[ _side * _side ];

        for ( unsigned i=0; i< _side*_side; ++i ) {
                _matrix[i] = 1.0/ ( _side*_side );
        }
}

template< typename InputElementType >
bool
SimpleConvolutionImageFilter< Image< InputElementType, 3 > >
::ProcessSlice (
        const Image< InputElementType, 3 > 	&in,
        Image< InputElementType, 3 >		&out,
        size_t			x1,
        size_t			y1,
        size_t			x2,
        size_t			y2,
        size_t			slice
)
{
        if ( !this->CanContinue() ) { //Someone wants filter to stop.
                return false;
        }
        int hside = _side/2;
        for ( size_t i = x1 + hside; i < x2 - hside; ++i ) {
                for ( size_t j = y1 + hside; j < y2 - hside; ++j ) {
                        InputElementType pom = 0;
                        for ( int mi = -hside; mi <= hside; ++ mi ) {
                                for ( int mj = -hside; mj <= hside; ++ mj ) {
                                        pom +=  in.GetElement ( i+mi, j+mj, slice ) * 1.0/25.0/*_matrix[ (mi + hside)* _side + mj + hside ]*/;
                                }
                        }

                        out.GetElement ( i, j, slice ) = pom;
                }
        }
        return true;
}

} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*_BASIC_IMAGE_FILTERS_H*/

/** @} */

