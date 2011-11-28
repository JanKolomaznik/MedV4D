/**
 * @ingroup imaging
 * @author Jan Kolomaznik
 * @file AImageFilterWholeAtOnce.tcc
 * @{
 **/

#ifndef _ABSTRACT_IMAGE_FILTER_WHOLEATONCE_H
#error File AImageFilterWholeAtOnce.tcc cannot be included directly!
#else

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging {

template< typename InputImageType, typename OutputImageType >
AImageFilterWholeAtOnce< InputImageType, OutputImageType >
::AImageFilterWholeAtOnce ( typename AImageFilterWholeAtOnce< InputImageType, OutputImageType >::Properties *prop )
                : PredecessorType ( prop )
{

}

template< typename InputImageType, typename OutputImageType >
bool
AImageFilterWholeAtOnce< InputImageType, OutputImageType >
::ExecutionThreadMethod ( APipeFilter::UPDATE_TYPE utype )
{
        utype = utype;
        D_BLOCK_COMMENT ( "++++ Entering ExecutionThreadMethod() - AImageFilterWholeAtOnce", "----- Leaving MainExecutionThread() - AImageFilterWholeAtOnce" );
        if ( ! ( _readerBBox->WaitWhileDirty() == MS_MODIFIED ) ) {
                _writerBBox->SetState ( MS_CANCELED );
                return false;
        }

        if ( ProcessImage ( * ( this->in ), * ( this->out ) ) ) {
                _writerBBox->SetModified();
                return true;
        } else {
                _writerBBox->SetState ( MS_CANCELED );
                return false;
        }
}

template< typename InputImageType, typename OutputImageType >
void
AImageFilterWholeAtOnce< InputImageType, OutputImageType >
::BeforeComputation ( APipeFilter::UPDATE_TYPE &utype )
{
        PredecessorType::BeforeComputation ( utype );

        //This kind of filter computes always on whole dataset
        utype = APipeFilter::RECALCULATION;
}

template< typename InputImageType, typename OutputImageType >
void
AImageFilterWholeAtOnce< InputImageType, OutputImageType >
::MarkChanges ( APipeFilter::UPDATE_TYPE utype )
{
        utype = utype;

        _readerBBox = this->in->GetWholeDirtyBBox(); //ApplyReaderBBox( *(this->in) );
        _writerBBox = & ( this->out->SetWholeDirtyBBox() ); //&(ApplyWriterBBox( *(this->out) ) );
}

template< typename InputImageType, typename OutputImageType >
void
AImageFilterWholeAtOnce< InputImageType, OutputImageType >
::AfterComputation ( bool successful )
{
        _readerBBox = ReaderBBoxInterface::Ptr();
        _writerBBox = NULL;

        PredecessorType::AfterComputation ( successful );
}
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
/*template< typename ElementType, unsigned dim >
ReaderBBoxInterface::Ptr
ApplyReaderBBox( const Image< ElementType, dim > &in );

template< typename ElementType, unsigned dim >
ReaderBBoxInterface::Ptr
ApplyReaderBBoxFunc( const Image< ElementType, 2 > &in )
{
	return in.GetDirtyBBox(
				in.GetDimensionExtents( 0 ).minimum,
				in.GetDimensionExtents( 1 ).minimum,
				in.GetDimensionExtents( 0 ).maximum,
				in.GetDimensionExtents( 1 ).maximum
			);
}

template< typename ElementType, unsigned dim >
ReaderBBoxInterface::Ptr
ApplyReaderBBoxFunc( const Image< ElementType, 3 > &in )
{
	return in.GetDirtyBBox(
				in.GetDimensionExtents( 0 ).minimum,
				in.GetDimensionExtents( 1 ).minimum,
				in.GetDimensionExtents( 2 ).minimum,
				in.GetDimensionExtents( 0 ).maximum,
				in.GetDimensionExtents( 1 ).maximum,
				in.GetDimensionExtents( 2 ).maximum
			);
}

template< typename ElementType, unsigned dim >
ReaderBBoxInterface::Ptr
ApplyReaderBBoxFunc( const Image< ElementType, 4 > &in )
{
	return in.GetDirtyBBox(
				in.GetDimensionExtents( 0 ).minimum,
				in.GetDimensionExtents( 1 ).minimum,
				in.GetDimensionExtents( 2 ).minimum,
				in.GetDimensionExtents( 3 ).minimum,
				in.GetDimensionExtents( 0 ).maximum,
				in.GetDimensionExtents( 1 ).maximum,
				in.GetDimensionExtents( 2 ).maximum,
				in.GetDimensionExtents( 3 ).maximum
			);
}

template< typename ElementType, unsigned dim >
WriterBBoxInterface &
ApplyWriterBBoxFunc( Image< ElementType, dim > &out );

template< typename ElementType, unsigned dim >
WriterBBoxInterface &
ApplyWriterBBoxFunc( Image< ElementType, 2 > &out )
{
	return out.SetDirtyBBox(
				out.GetDimensionExtents( 0 ).minimum,
				out.GetDimensionExtents( 1 ).minimum,
				out.GetDimensionExtents( 0 ).maximum,
				out.GetDimensionExtents( 1 ).maximum
			);
}

template< typename ElementType, unsigned dim >
WriterBBoxInterface &
ApplyWriterBBoxFunc( Image< ElementType, 3 > &out )
{
	return out.SetDirtyBBox(
				out.GetDimensionExtents( 0 ).minimum,
				out.GetDimensionExtents( 1 ).minimum,
				out.GetDimensionExtents( 2 ).minimum,
				out.GetDimensionExtents( 0 ).maximum,
				out.GetDimensionExtents( 1 ).maximum,
				out.GetDimensionExtents( 2 ).maximum
			);
}

template< typename ElementType, unsigned dim >
WriterBBoxInterface &
ApplyWriterBBoxFunc( Image< ElementType, 4 > &out )
{
	return out.SetDirtyBBox(
				out.GetDimensionExtents( 0 ).minimum,
				out.GetDimensionExtents( 1 ).minimum,
				out.GetDimensionExtents( 2 ).minimum,
				out.GetDimensionExtents( 3 ).minimum,
				out.GetDimensionExtents( 0 ).maximum,
				out.GetDimensionExtents( 1 ).maximum,
				out.GetDimensionExtents( 2 ).maximum,
				out.GetDimensionExtents( 3 ).maximum
			);
}*/
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
/*template< typename InputImageType, typename OutputImageType >
ReaderBBoxInterface::Ptr
AImageFilterWholeAtOnce< InputImageType, OutputImageType >
::ApplyReaderBBox( const InputImageType &in )
{
	return ApplyReaderBBoxFunc< InputImageType::Element, InputImageType::Dimension >( in );
}

template< typename InputImageType, typename OutputImageType >
WriterBBoxInterface &
AImageFilterWholeAtOnce< InputImageType, OutputImageType >
::ApplyWriterBBox( OutputImageType &out )
{
	return ApplyWriterBBoxFunc< OutputImageType::Element, OutputImageType::Dimension >( out );
}*/

//*****************************************************************************
//*****************************************************************************
//*****************************************************************************


template< typename InputImageType, typename OutputImageType >
AImageFilterWholeAtOnceIExtents< InputImageType, OutputImageType >
::AImageFilterWholeAtOnceIExtents ( typename AImageFilterWholeAtOnceIExtents< InputImageType, OutputImageType >::Properties *prop )
                : PredecessorType ( prop )
{

}

template< typename InputImageType, typename OutputImageType >
void
AImageFilterWholeAtOnceIExtents< InputImageType, OutputImageType >
::PrepareOutputDatasets()
{
        PredecessorType::PrepareOutputDatasets();

        int32 minimums[ ImageTraits<InputImageType>::Dimension ];
        int32 maximums[ ImageTraits<InputImageType>::Dimension ];
        float32 voxelExtents[ ImageTraits<InputImageType>::Dimension ];

        for ( unsigned i=0; i <  ImageTraits<InputImageType>::Dimension; ++i ) {
                const DimensionExtents & dimExt = this->in->GetDimensionExtents ( i );

                minimums[i] = dimExt.minimum;
                maximums[i] = dimExt.maximum;
                voxelExtents[i] = dimExt.elementExtent;
        }
        this->SetOutputImageSize ( minimums, maximums, voxelExtents );
}

} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*_ABSTRACT_IMAGE_FILTER_WHOLEATONCE_H*/


/** @} */

