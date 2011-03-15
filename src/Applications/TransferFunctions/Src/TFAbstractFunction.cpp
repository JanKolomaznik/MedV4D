#include <TFAbstractFunction.h>


namespace M4D {
namespace GUI {

TFAbstractFunction::TFAbstractFunction():
	type_(TFFUNCTION_UNKNOWN){
}

TFAbstractFunction::~TFAbstractFunction(){}

TFFunctionType TFAbstractFunction::getType() const{

	return type_;
}

const TFSize TFAbstractFunction::getDomain(){

	return domain_;
}

TFColorMapPtr TFAbstractFunction::getColorMap(){

	return colorMap_;
}

void TFAbstractFunction::clear(){

	TFColorMapIt begin = colorMap_->begin();
	TFColorMapIt end = colorMap_->end();
	for(TFColorMapIt it = begin; it!=end; ++it)
	{
		it->component1 = 0;
		it->component2 = 0;
		it->component3 = 0;
		it->alpha = 0;
	}
}

void TFAbstractFunction::resize(const TFSize domain){
	
	if(domain == domain_) return;
	domain_ = domain;

	const TFColorMapPtr old = colorMap_;
	TFColorMapPtr resized = TFColorMapPtr(new TFColorMap(domain_));

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
			TFSize valueCount = ratio + (int)correction;
			for(TFSize i = 0; i < valueCount; ++i)
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
			TFColor computedValue(0,0,0,0);
			TFSize valueCount = ratio + (int)correction;
			for(TFSize i = 0; i < valueCount; ++i)
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

void TFAbstractFunction::operator=(TFAbstractFunction &function){

	type_ = function.getType();
	
	colorMap_->clear();

	const TFColorMapPtr colorMap = function.getColorMap();

	TFColorMap::const_iterator begin = colorMap->begin();
	TFColorMap::const_iterator end = colorMap->end();
	for(TFColorMap::const_iterator it = begin; it!=end; ++it)
	{
		colorMap_->push_back(*it);
	}
}

} // namespace GUI
} // namespace M4D