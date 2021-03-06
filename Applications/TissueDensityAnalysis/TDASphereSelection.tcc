/**
 * @author Milan Lepik
 * @file TDASphereSelection.tcc 
 * @{ 
 **/

#ifndef _TDA_SPHERE_SELECTION_H
#error File TDASphereSelection.tcc cannot be included directly!
#else

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{

namespace Imaging
{


template< typename ElementType >
TDASphereSelection< Image< ElementType, 3 > >
::TDASphereSelection() : PredecessorType( new Properties() )
{

}

template< typename ElementType >
TDASphereSelection< Image< ElementType, 3 > >
::TDASphereSelection( typename TDASphereSelection< Image< ElementType, 3 > >::Properties *prop ) 
: PredecessorType( prop ) 
{

}

template< typename ElementType >
void
TDASphereSelection< Image< ElementType, 3 > >
::BeforeComputation( APipeFilter::UPDATE_TYPE &utype )
{
	PredecessorType::BeforeComputation( utype );

	if( this->_propertiesTimestamp != GetProperties().GetTimestamp() )
	{
		utype = APipeFilter::RECALCULATION;
		this->_callPrepareOutputDatasets = true;
	}
}

template< typename ElementType >
bool
TDASphereSelection< Image< ElementType, 3 > >
::ProcessImage(
			const Image< ElementType, 3 > 	&in,
			Image< ElementType, 3 >		&out
		    )
{
	return ProcessImageHelper< AverageOperator< ElementType > >( in, out );
}

template< typename ElementType >
template< typename OperatorType >
bool
TDASphereSelection< Image< ElementType, 3 > >
::ProcessImageHelper(
			const Image< ElementType, 3 > 	&in,
			Image< ElementType, 3 >		&out
		    )
{

	Vector< uint32, 3 > size1;
	Vector< int32, 3 > strides1;
	ElementType *pointer1 = in.GetPointer( size1, strides1 );

	Vector< uint32, 3 > size2;
	Vector< int32, 3 > strides2;
	ElementType *pointer2 = out.GetPointer( size2, strides2 );

	return DoProjection< OperatorType >(
				pointer1,
				pointer2,
				strides1[0],
				strides1[1],
				strides1[2],
				strides2[0],
				strides2[1],
				strides2[2],
				size1[0],
				size1[1],
				size1[2]
			    );
}

template< typename ElementType >
void
TDASphereSelection< Image< ElementType, 3 > >
::PrepareOutputDatasets()
{
	PredecessorType::PrepareOutputDatasets();

	int32 minimums[ 3 ];
	int32 maximums[ 3 ];
	float32 voxelExtents[ 3 ];

	minimums[0] = this->in->GetDimensionExtents( 0 ).minimum;
	maximums[0] = this->in->GetDimensionExtents( 0 ).maximum;
	voxelExtents[0] = this->in->GetDimensionExtents( 0 ).elementExtent;

	minimums[1] = this->in->GetDimensionExtents( 1 ).minimum;
	maximums[1] = this->in->GetDimensionExtents( 1 ).maximum;
	voxelExtents[1] = this->in->GetDimensionExtents( 1 ).elementExtent;

	minimums[2] = this->in->GetDimensionExtents( 2 ).minimum;
	maximums[2] = this->in->GetDimensionExtents( 2 ).maximum;
	voxelExtents[2] = this->in->GetDimensionExtents( 2 ).elementExtent;

	this->SetOutputImageSize( minimums, maximums, voxelExtents );
}

template< typename ElementType >
template< typename OperatorType >
bool
TDASphereSelection< Image< ElementType, 3 > >
::DoProjection(
			ElementType	*inPointer,
			ElementType	*outPointer,
			int32		ixStride,
			int32		iyStride,
			int32		izStride,
			int32		oxStride,
			int32		oyStride,
			int32		ozStride,
			uint32		width,
			uint32		height,
			uint32		depth
		    )
{
	if( !this->CanContinue() ) {
		return false;
	}
	

	uint32 radius = GetProperties().radius;
	uint32 radiusPow = pow((double)radius, 2);

	//center of circle
	int32 iCenter = GetProperties().iCenter;
	int32 jCenter = GetProperties().jCenter;
	int32 kCenter = GetProperties().kCenter;

	double toSliceSize = 0.5; // pomer vysky voxelu k jeho sirce(hloubce)
	double toVoxelSize = 1/toSliceSize; // pomer vysky voxelu k jeho sirce(hloubce)

	//inicializace masky
	ElementType *inSlicePointer = inPointer;
	ElementType *outSlicePointer = outPointer;
	for( uint32 k = 0; k < depth; ++k ) {
		ElementType *outRowPointer = outSlicePointer;
		for( uint32 j = 0; j < height; ++j ) {
			ElementType *outColPointer = outRowPointer;
			for( uint32 i = 0; i < width; ++i ) {
				*outColPointer = 0;
				outColPointer += oxStride;
			}
			outRowPointer += oyStride;
		}
		outSlicePointer += ozStride;
	}
	

	//square tresholds of area where the sphere is located
	//zjistime na ktere rezy maska protne
	double zRadius = radius * toSliceSize; //polomer protnutych rezu
	int32 kMin = std::max(0.0, kCenter - zRadius); //nejnizsi cislo rezu
	int32 kMax = std::min(double(depth), kCenter + zRadius); //nejvyssi cislo rezu

	uint32 zDiameter = kMax - kMin; //celocislny pocet protnutych rezu
	double zRadiusPow = pow(zRadius, 2); // predpocitame si druhou mocninu at to pak nepocitame v cyklu pokazde znova
	inSlicePointer = inPointer + kMin * izStride; // ukazatel na prvni rez ve vstupni masce
	outSlicePointer = outPointer + kMin * ozStride; // ukazatel na prvni rez ve vystupni masce
	for( int32 k = kMin; k < kMax; ++k ) { //pro kazdy rez protinajici masku
	
		int32 high = abs(k-kMin); //vyska od spodniho rezu
		float actRadiusPow = zRadiusPow - pow(zRadius - high, 2 ) * toVoxelSize; // druha mocnina polomeru kruhu v aktualnim rezu
		float actRadius = sqrt(actRadiusPow) * toVoxelSize; //polomer kruhu v aktualnim rezu


		//square tresholds of area where the sphere is located - according to actual slice
		int32 iMin = std::max(0,iCenter-(int)actRadius);
		int32 iMax = std::min((int)width,iCenter+(int)actRadius);
		int32 jMin = std::max(0,jCenter-(int)actRadius);
		int32 jMax = std::min((int)height,jCenter+(int)actRadius);

		ElementType *inRowPointer = inSlicePointer+jMin*iyStride;
		ElementType *outRowPointer = outSlicePointer+jMin*oyStride;
		for( int32 j = jMin; j < jMax; ++j ) {

			ElementType *inColPointer = inRowPointer+iMin*ixStride;
			ElementType *outColPointer = outRowPointer+iMin*oxStride;
			for( int32 i = iMin; i < iMax; ++i ) {
				if (actRadiusPow >= pow((double)(iCenter - i), 2)+pow((double)(jCenter - j), 2))
					*outColPointer = *inColPointer;
				else
					*outColPointer = 0;
				
				inColPointer += ixStride;
				outColPointer += oxStride;
			}

			inRowPointer += iyStride;
			outRowPointer += oyStride;
		}

		inSlicePointer += izStride;
		outSlicePointer += ozStride;
	}
	
	return true;
}

} /*namespace Imaging*/
} /*namespace M4D*/

/** @} */

#endif /*_SPHERE_SELECTION_H*/

/** @} */

