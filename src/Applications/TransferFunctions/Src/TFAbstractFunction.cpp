#include <TFAbstractFunction.h>


namespace M4D {
namespace GUI {
/*
template<TF::Size dim>
TFAbstractFunction<dim>::TFAbstractFunction(){}

template<TF::Size dim>
TFAbstractFunction<dim>::~TFAbstractFunction(){}

template<TF::Size dim>
TF::Size TFAbstractFunction<dim>::getDimension(){

	return dim;
}

template<TF::Size dim>
TF::MultiDColor<dim>& TFAbstractFunction<dim>::operator[](const TF::Size index){

	return (*colorMap_)[index];
}

template<TF::Size dim>
TF::Size TFAbstractFunction<dim>::getDomain() const{

	return domain_;
}

TF::MultiDColor<dim>::Map::Ptr TFAbstractFunction<dim>::getColorMap(){

	return colorMap_;
}

template<TF::Size dim>
void TFAbstractFunction<dim>::clear(){

	TF::MultiDColor<dim>::Map::iterator begin = colorMap_->begin();
	TF::MultiDColor<dim>::Map::iterator end = colorMap_->end();
	for(TF::MultiDColor<dim>::Map::iterator it = begin; it!=end; ++it)
	{
		*it = TF::MultiDColor<dim>();
 	}
}

template<TF::Size dim>
void TFAbstractFunction<dim>::resize(const TF::Size domain){
	
	if(domain == domain_) return;
	domain_ = domain;

	const TF::MultiDColor<dim>::Map::Ptr old = colorMap_;
	TF::MultiDColor<dim>::Map::Ptr resized(new TF::MultiDColor<dim>::Map(domain_));

	int inputSize = old->size();
	int outputSize = resized->size();

	float correction = outputSize/(float)inputSize;

	if(correction >= 1)
	{
		int ratio = (int)(correction);	//how many old values are used for computing 1 resized values
		correction -= ratio;
		float corrStep = correction;

		int outputIndexer = 0;
		for(int inputIndexer = 0; inputIndexer < inputSize; ++inputIndexer)
		{
			TF::Size valueCount = ratio + (int)correction;
			for(TF::Size i = 0; i < valueCount; ++i)
			{
				//tfAssert(outputIndexer < outputSize);
				if(inputIndexer >= inputSize) break;

				(*resized)[outputIndexer] = (*old)[inputIndexer];

				++outputIndexer;
			}
			correction -= (int)correction;
			correction += corrStep;
		}
	}
	else
	{
		correction = inputSize/(float)outputSize;
		int ratio =  (int)(correction);	//how many old values are used for computing 1 resized values
		correction -= ratio;
		float corrStep = correction;

		int inputIndexer = 0;
		for(int outputIndexer = 0; outputIndexer < outputSize; ++outputIndexer)
		{
			TF::MultiDColor computedValue(dimension_);
			TF::Size valueCount = ratio + (int)correction;
			for(TF::Size i = 0; i < valueCount; ++i)
			{
				//tfAssert(inputIndexer < inputSize);
				if(inputIndexer >= inputSize)
				{
					valueCount = i;
					break;
				}

				computedValue += (*old)[inputIndexer];

				++inputIndexer;
			}
			correction -= (int)correction;
			correction += corrStep;

			if(valueCount == 0) break;
			(*resized)[outputIndexer] = computedValue/valueCount;
		}
	}

	colorMap_ = resized;
}

template<TF::Size dim>
void TFAbstractFunction<dim>::operator=(const TFAbstractFunction<dim> &function){
	
	*colorMap_ = *function.colorMap_;
	domain_ = function.domain_;
}
*/
} // namespace GUI
} // namespace M4D