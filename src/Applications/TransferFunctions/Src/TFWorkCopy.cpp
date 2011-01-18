#include <TFWorkCopy.h>


namespace M4D {
namespace GUI {

TFWorkCopy::TFWorkCopy(const TFSize& domain):
	domain_(domain),
	data_(new TFColorMap()),
	xSize_(0),
	ySize_(0){
}

//---getters---

TFColor TFWorkCopy::getColor(const TFSize& index){

	TFColor fullColor(
		(*data_)[index].component1/zoom_.zoom + zoom_.yOffset,
		(*data_)[index].component2/zoom_.zoom + zoom_.yOffset,
		(*data_)[index].component3/zoom_.zoom + zoom_.yOffset,
		1);
	return fullColor;
}

float TFWorkCopy::getComponent1(const TFSize& index){

	return (*data_)[index].component1;
}

float TFWorkCopy::getComponent2(const TFSize& index){

	return (*data_)[index].component2;
}

float TFWorkCopy::getComponent3(const TFSize& index){

	return (*data_)[index].component3;
}

float TFWorkCopy::getAlpha(const TFSize& index){

	return (*data_)[index].alpha;
}

//---setters---

void TFWorkCopy::setComponent1(const TFSize& index, const float& value){

	(*data_)[index].component1 = value;
}

void TFWorkCopy::setComponent2(const TFSize& index, const float& value){

	(*data_)[index].component2 = value;
}

void TFWorkCopy::setComponent3(const TFSize& index, const float& value){

	(*data_)[index].component3 = value;
}

void TFWorkCopy::setAlpha(const TFSize& index, const float& value){

	(*data_)[index].alpha = value;
}

//---size---

void TFWorkCopy::resize(const TFSize& xSize, const TFSize& ySize){

	xSize_ = xSize;
	ySize_ = ySize;
	data_->resize(xSize_);
}

TFSize TFWorkCopy::size(){

	return data_->size();
}

//---zoom---

void TFWorkCopy::zoomIn(const TFSize& stepCount, const TFSize& inputX, const TFSize& inputY){

	float nextZoom = zoom_.zoom;
	for(TFSize i = 0; i < stepCount; ++i)
	{
		nextZoom *= 2;
		if(data_->size()*nextZoom > domain_)
		{
			nextZoom /= 2;
			break;
		}
	}
	if(nextZoom == zoom_.zoom) return;
	computeZoom_(nextZoom, inputX, inputY);
}

void TFWorkCopy::zoomOut(const TFSize& stepCount, const TFSize& inputX, const TFSize& inputY){

	if(zoom_.zoom == 1) return;

	float nextZoom = zoom_.zoom;
	for(TFSize i = 0; i < stepCount; ++i)
	{
		nextZoom /= 2;
		if(nextZoom == 1)
		{
			zoom_.reset();
			return;
		}
	}
	computeZoom_(nextZoom, inputX, inputY);
}

void TFWorkCopy::move(int xDirectionIncrement, int yDirectionIncrement){


	TFSize moveX = xSize_/2 + xDirectionIncrement;
	TFSize moveY = ySize_/2 + yDirectionIncrement;

	computeZoom_(zoom_.zoom, moveX, moveY);
}

const float& TFWorkCopy::zoom(){

	return zoom_.zoom;
}

//---update---

void TFWorkCopy::updateFunction(TFAbstractFunction::Ptr function){

	const TFColorMapPtr input = data_;
	TFColorMapPtr output = function->getColorMap();

	TFSize inputSize = input->size();
	TFSize outputSize = output->size();

	float correction = outputSize/(inputSize*zoom_.zoom);
	int ratio = (int)(correction);	//how many input values are used for computing 1 output values
	correction -= ratio;
	float corrStep = correction;

	TFSize outputIndexer = 0;
	for(TFSize inputIndexer = 0; inputIndexer < inputSize; ++inputIndexer)
	{
		TFSize valueCount = ratio + (int)correction;
		for(TFSize i = 0; i < valueCount; ++i)
		{
			tfAssert((outputIndexer + zoom_.xOffset) < outputSize);

			(*output)[outputIndexer + zoom_.xOffset].component1 =
				(*input)[inputIndexer].component1/zoom_.zoom + zoom_.yOffset;

			(*output)[outputIndexer + zoom_.xOffset].component2 =
				(*input)[inputIndexer].component2/zoom_.zoom + zoom_.yOffset;

			(*output)[outputIndexer + zoom_.xOffset].component3 =
				(*input)[inputIndexer].component3/zoom_.zoom + zoom_.yOffset;

			(*output)[outputIndexer + zoom_.xOffset].alpha =
				(*input)[inputIndexer].alpha/zoom_.zoom + zoom_.yOffset;

			++outputIndexer;
		}
		correction -= (int)correction;
		correction += corrStep;
	}
}

void TFWorkCopy::update(TFAbstractFunction::Ptr function){
	
	const TFColorMapPtr input = function->getColorMap();
	TFColorMapPtr output = data_;

	TFSize inputSize = input->size();
	TFSize outputSize = output->size();

	float correction = inputSize/(outputSize*zoom_.zoom);
	int ratio =  (int)(correction);	//how many input values are used for computing 1 output values
	correction -= ratio;
	float corrStep = correction;

	float increment = 0;
	TFSize inputIndexer = 0;
	for(TFSize outputIndexer = 0; outputIndexer < outputSize; ++outputIndexer)
	{
		TFColor computedValue(0,0,0,0);
		TFSize valueCount = ratio + (int)correction;
		for(TFSize i = 0; i < valueCount; ++i)
		{
			tfAssert(inputIndexer + zoom_.xOffset < inputSize);

			increment = ((*input)[inputIndexer + zoom_.xOffset].component1 - zoom_.yOffset)*zoom_.zoom;
			computedValue.component1 += increment;

			increment = ((*input)[inputIndexer + zoom_.xOffset].component2 - zoom_.yOffset)*zoom_.zoom;
			computedValue.component2 += increment;

			increment = ((*input)[inputIndexer + zoom_.xOffset].component3 - zoom_.yOffset)*zoom_.zoom;
			computedValue.component3 += increment;

			increment = ((*input)[inputIndexer + zoom_.xOffset].alpha - zoom_.yOffset)*zoom_.zoom;
			computedValue.alpha += increment;

			++inputIndexer;
		}
		correction -= (int)correction;
		correction += corrStep;

		computedValue.component1 = computedValue.component1/valueCount;
		computedValue.component2 = computedValue.component2/valueCount;
		computedValue.component3 = computedValue.component3/valueCount;
		computedValue.alpha = computedValue.alpha/valueCount;

		(*output)[outputIndexer] = computedValue;
	}
}

void TFWorkCopy::computeZoom_(const float& nextZoom, const TFSize& inputX, const TFSize& inputY){

	float relativeZoomedRatioX = (domain_/zoom_.zoom)/xSize_;
	float relativeZoomedRatioY = (1/zoom_.zoom)/ySize_;

	int xRadius = (int)(domain_/nextZoom/2.0f);
	int xOffesetInc = zoom_.xOffset + (int)(inputX*relativeZoomedRatioX) - xRadius;

	int maxXOffset = domain_ - 2*xRadius;
	
	if(xOffesetInc < 0) xOffesetInc = 0;
	if(xOffesetInc > maxXOffset) xOffesetInc = maxXOffset;

	float yRadius = 1/nextZoom/2.0f;
	float yOffesetInc = zoom_.yOffset + inputY*relativeZoomedRatioY - yRadius;

	float maxYOffset = 1 - 2*yRadius;

	if(yOffesetInc < 0) yOffesetInc = 0;
	if(yOffesetInc > maxYOffset) yOffesetInc = maxYOffset;

	zoom_.zoom = nextZoom;
	zoom_.xOffset = xOffesetInc;
	zoom_.yOffset = yOffesetInc;
}

} // namespace GUI
} // namespace M4D