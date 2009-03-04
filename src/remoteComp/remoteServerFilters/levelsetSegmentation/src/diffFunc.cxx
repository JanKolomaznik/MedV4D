#ifndef CELLTHRESHOLDLEVELSETFINITEDIFFERENCEFUNCTION_H_
#error File diffFunc.cxx cannot be included directly!
#else

namespace itk
{

///////////////////////////////////////////////////////////////////////////////

template <class TImageType, class TFeatureImageType>
ThresholdLevelSetFunc<TImageType, TFeatureImageType>
::ThresholdLevelSetFunc()
{
	RadiusType radius;
	radius[0] = radius[1] = radius[2] = 1; 
	this->SetRadius(radius);
	
	cntr_.Reset();
}

///////////////////////////////////////////////////////////////////////////////

template <class TImageType, class TFeatureImageType>
typename ThresholdLevelSetFunc<TImageType, TFeatureImageType>::PixelType
ThresholdLevelSetFunc<TImageType, TFeatureImageType>
	::ComputeUpdate(const NeighborhoodType &neighborhood, void *globalData,
            const FloatOffsetType& offset)
{
	typedef LevelSetFunction<TImageType> LSFunc;
	typedef typename LSFunc::NeighborhoodType LSNeighborhoodType;
	typedef typename LSFunc::FloatOffsetType LSFloatOffsetType;
	
	cntr_.Start();
	PixelType retval = LevelSetFunction<TImageType>::ComputeUpdate(
			(const LSNeighborhoodType &) neighborhood,
			globalData,
			(const LSFloatOffsetType &) offset);
	cntr_.Stop();
	
	return retval;
}

///////////////////////////////////////////////////////////////////////////////
}
#endif