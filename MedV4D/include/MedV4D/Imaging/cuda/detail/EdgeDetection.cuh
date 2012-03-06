#ifndef EDGE_DETECTION_CUH
#define EDGE_DETECTION_CUH

template< typename TElement >
struct SobelFilter3DFtor
{
	typedef typename TypeTraits< TElement >::SignedClosestType SignedElement;

	SobelFilter3DFtor( TElement aThreshold = 0 ): threshold( aThreshold ), radius( make_int3( 1, 1, 1 ) )
	{}

	__device__ TElement
	operator()( TElement data[], uint idx, uint syStride, uint szStride, uint gIdx )
	{
		TElement val1 = abs( (2*static_cast<SignedElement>(data[idx+1]) - 2*static_cast<SignedElement>(data[idx-1])) 
			+ (static_cast<SignedElement>(data[idx+1 + syStride]) - static_cast<SignedElement>(data[idx-1 + syStride]))
			+ (static_cast<SignedElement>(data[idx+1 - syStride]) - static_cast<SignedElement>(data[idx-1 - syStride])) );
		TElement val2 = abs( (2*static_cast<SignedElement>(data[idx+syStride]) - 2*static_cast<SignedElement>(data[idx-syStride])) 
			+ (static_cast<SignedElement>(data[idx+1 + syStride]) - static_cast<SignedElement>(data[idx+1 - syStride]))
			+ (static_cast<SignedElement>(data[idx-1 + syStride]) - static_cast<SignedElement>(data[idx-1 - syStride])) );
		TElement val3 = abs( (2*static_cast<SignedElement>(data[idx+szStride]) - 2*static_cast<SignedElement>(data[idx-szStride])) 
			+ (static_cast<SignedElement>(data[idx + syStride + szStride]) - static_cast<SignedElement>(data[idx + syStride - szStride]))
			+ (static_cast<SignedElement>(data[idx - syStride + szStride]) - static_cast<SignedElement>(data[idx - syStride - szStride])) );
		TElement val4 = abs( (2*static_cast<SignedElement>(data[idx + syStride]) - 2*static_cast<SignedElement>(data[idx-syStride])) 
			+ (static_cast<SignedElement>(data[idx + syStride + szStride]) - static_cast<SignedElement>(data[idx - syStride + szStride]))
			+ (static_cast<SignedElement>(data[idx + syStride - szStride]) - static_cast<SignedElement>(data[idx - syStride - szStride])) );
		TElement val5 = abs( (2*static_cast<SignedElement>(data[idx+1]) - 2*static_cast<SignedElement>(data[idx-1])) 
			+ (static_cast<SignedElement>(data[idx + 1 + szStride]) - static_cast<SignedElement>(data[idx-1 + szStride]))
			+ (static_cast<SignedElement>(data[idx + 1 - szStride]) - static_cast<SignedElement>(data[idx-1 - szStride])) );
		TElement val6 = abs( (2*static_cast<SignedElement>(data[idx+szStride]) - 2*static_cast<SignedElement>(data[idx-szStride])) 
			+ (static_cast<SignedElement>(data[idx + 1 + szStride]) - static_cast<SignedElement>(data[idx +1 - szStride]))
			+ (static_cast<SignedElement>(data[idx - 1 + szStride]) - static_cast<SignedElement>(data[idx -1 - szStride])) );
		TElement result = val1 + val2 + val3 + val4 + val5 +val6;
		return result > threshold ? result : 0;
	}

	TElement threshold;
	int3 radius;
};

#endif //EDGE_DETECTION_CUH