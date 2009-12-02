#ifndef DECIMATION_H_
#define DECIMATION_H_

#include "common/Common.h"
#include "Imaging/AImageFilterWholeAtOnce.h"

namespace M4D
{

/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file MedianFilter.h 
 * @{ 
 **/

namespace Imaging
{

template< typename ImageType, typename InterpolatorType>
class DecimationFilter
	: public AImageFilterWholeAtOnce< ImageType, ImageType >
{
public:	
	typedef DecimationFilter<ImageType, InterpolatorType> Self;
	typedef AImageFilterWholeAtOnce< ImageType, ImageType > PredecessorType;
	typedef typename ImageTraits< ImageType >::ElementType ElementType;

	struct Properties : public PredecessorType::Properties
	{
		Properties(): ratio( 1 ) {}
		Properties(float32	ratio_): ratio( ratio_) {}
		
		float32	ratio;
	};

	DecimationFilter( Properties * prop );

	GET_SET_PROPERTY_METHOD_MACRO( float32, Ratio, ratio );
protected:

	void PrepareOutputDatasets(void);
	bool ProcessImage(const ImageType &in, ImageType &out);

private:

	GET_PROPERTIES_DEFINITION_MACRO;
	
	static void Process3DImage(Self *self);

};


} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "decimation.tcc"

#endif /*DECIMATION_H_*/
