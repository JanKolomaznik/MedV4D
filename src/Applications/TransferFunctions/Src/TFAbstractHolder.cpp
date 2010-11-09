#include <TFAbstractHolder.h>

namespace M4D {
namespace GUI {

TFAbstractHolder::TFAbstractHolder():
	type_(TFTYPE_UNKNOWN),
	basicTools_(new Ui::TFAbstractHolder),
	setup_(false){

	basicTools_->setupUi(this);
}

TFAbstractHolder::~TFAbstractHolder(){
	if(basicTools_) delete basicTools_;
}

void TFAbstractHolder::save(){

	QString fileName = QFileDialog::getSaveFileName(this,
		tr("Save Transfer Function"),
		QDir::currentPath(),
		tr("TF Files (*.tf)"));

	if (fileName.isEmpty()) return;

	QFile file(fileName);
	if (!file.open(QFile::WriteOnly | QFile::Text))
	{
		QMessageBox::warning(this, tr("Transfer Functions"),
			tr("Cannot write file %1:\n%2.")
			.arg(fileName)
			.arg(file.errorString()));
		return;
	}

	save_(file);

	file.close();
}

TFType TFAbstractHolder::getType() const{
	return type_;
}

void TFAbstractHolder::size_changed(const QRect rect){
	
	setGeometry(rect);

	updateFunction_();	

	int newWidth = rect.width() - 2*PAINTER_X;
	int newHeight = rect.height() - 2*PAINTER_Y;
	QRect newRect(PAINTER_X, PAINTER_Y, newWidth, newHeight);

	updatePainter_(newRect);
}

void TFAbstractHolder::calculate_(const TFFunctionMapPtr input, TFFunctionMapPtr output){

	if(!(input && output)) tfAbort("calculation error");
	if(output->begin() == output->end())
	{
		tfAssert(!"empty output for calculation");
		return;
	}
	if(input->begin() == input->end())
	{
		tfAssert(!"empty input for calculation");
		return;
	}

	TFSize inputSize = input->size();
	TFSize outputSize = output->size();
	float ratio = inputSize/(float)outputSize;

	if(ratio > 1)
	{
		float inOutCorrection = ratio;
		int inOutRatio =  (int)(inOutCorrection);	//how many input values are used for computing 1 output values
		inOutCorrection -= inOutRatio;
		float corrStep = inOutCorrection;

		TFFunctionMapIt outIt = output->begin();

		TFFunctionMap::const_iterator inBegin = input->begin();
		TFFunctionMap::const_iterator inEnd = input->end();

		for(TFFunctionMap::const_iterator it = inBegin; it != inEnd; ++it)
		{
			float computedValue = 0;
			TFSize valueCount = inOutRatio + (int)inOutCorrection;
			for(TFSize i = 0; i < valueCount; ++i)
			{
				if(it == inEnd) return;		//TODO fail
				computedValue += *it;
				if(i < (valueCount-1)) ++it;
			}
			inOutCorrection -= (int)inOutCorrection;
			inOutCorrection += corrStep;

			tfAssert(outIt != output->end());
			float avarage = computedValue/valueCount;
			*outIt = avarage;
			++outIt;
		}
	}
	else
	{
		float outInCorrection = outputSize/(float)inputSize;
		int outInRatio = (int)(outInCorrection);	//how many input values are used for computing 1 output values
		outInCorrection -= outInRatio;
		float corrStep = outInCorrection;

		TFFunctionMapIt outIt = output->begin();

		TFFunctionMap::const_iterator inBegin = input->begin();
		TFFunctionMap::const_iterator inEnd = input->end();

		for(TFFunctionMap::const_iterator it = inBegin; it != inEnd; ++it)
		{
			TFSize valueCount = outInRatio + (int)outInCorrection;
			for(TFSize i = 0; i < valueCount; ++i)
			{
				tfAssert(outIt != output->end());
				*outIt = *it;
				++outIt;
			}
			outInCorrection -= (int)outInCorrection;
			outInCorrection += corrStep;
		}
	}
}

} // namespace GUI
} // namespace M4D
