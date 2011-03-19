#include <TFAbstractFunction.h>


namespace M4D {
namespace GUI {

TFAbstractFunction::TFAbstractFunction(){}

TFAbstractFunction::~TFAbstractFunction(){}

TF::Color& TFAbstractFunction::operator[](const TF::Size index){

	return (*colorMap_)[index];
}

TF::Size TFAbstractFunction::getDomain() const{

	return domain_;
}
/*
TF::ColorMapPtr TFAbstractFunction::getColorMap(){

	return colorMap_;
}
*/
void TFAbstractFunction::clear(){

	TF::ColorMapIt begin = colorMap_->begin();
	TF::ColorMapIt end = colorMap_->end();
	for(TF::ColorMapIt it = begin; it!=end; ++it)
	{
		it->component1 = 0;
		it->component2 = 0;
		it->component3 = 0;
		it->alpha = 0;
	}
}

void TFAbstractFunction::resize(const TF::Size domain){
	
	if(domain == domain_) return;
	domain_ = domain;

	const TF::ColorMapPtr old = colorMap_;
	TF::ColorMapPtr resized = TF::ColorMapPtr(new TF::ColorMap(domain_));

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
			TF::Color computedValue(0,0,0,0);
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

void TFAbstractFunction::operator=(const TFAbstractFunction &function){
	
	colorMap_->clear();

	const TF::ColorMapPtr colorMap = function.colorMap_;

	TF::ColorMap::const_iterator begin = colorMap->begin();
	TF::ColorMap::const_iterator end = colorMap->end();
	for(TF::ColorMap::const_iterator it = begin; it!=end; ++it)
	{
		colorMap_->push_back(*it);
	}
}

} // namespace GUI
} // namespace M4D