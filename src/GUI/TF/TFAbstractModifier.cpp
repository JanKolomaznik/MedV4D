#include "GUI/TF/TFAbstractModifier.h"

namespace M4D {
namespace GUI {

TFAbstractModifier::TFAbstractModifier(TFFunctionInterface::Ptr function, TFAbstractPainter::Ptr painter):
	painter_(painter),
	workCopy_(TFWorkCopy::Ptr(new TFWorkCopy(function))),
	coords_(function->getDimension()),
	ignorePoint_(-1, -1),
	toolsWidget_(NULL),
	changed_(true){
}

QWidget* TFAbstractModifier::getTools(){

	if(!toolsWidget_) createTools_();
	return toolsWidget_;
}

TF::Size TFAbstractModifier::getDimension(){

	return workCopy_->getDimension();
}

TFFunctionInterface::Const TFAbstractModifier::getFunction(){

	return workCopy_->getFunction();
}

void TFAbstractModifier::setHistogram(const TF::HistogramInterface::Ptr histogram){

	if(histogram->getDimension() != workCopy_->getDimension()) workCopy_->setHistogram(TF::HistogramInterface::Ptr());
	else workCopy_->setHistogram(histogram);
}

bool TFAbstractModifier::changed(){

	if(changed_)
	{
		changed_ = false;
		return true;
	}
	return false;
}

M4D::Common::TimeStamp TFAbstractModifier::getTimeStamp(){

	return stamp_;
}

void TFAbstractModifier::save(TF::XmlWriterInterface* writer){

	painter_->save(writer);
	workCopy_->save(writer);
	saveSettings_(writer);
}

void TFAbstractModifier::saveFunction(TF::XmlWriterInterface* writer){

	workCopy_->saveFunction(writer);
}

bool TFAbstractModifier::load(TF::XmlReaderInterface* reader, bool& sideError){

	#ifndef TF_NDEBUG
		std::cout << "Loading modifier..." << std::endl;
	#endif

	bool painterOk = painter_->load(reader);
	bool workCopyOk = workCopy_->load(reader, sideError);

	bool error = !loadSettings_(reader);
	sideError = sideError || error;

	return painterOk && workCopyOk;
}

bool TFAbstractModifier::loadFunction(TF::XmlReaderInterface* reader){

	return workCopy_->loadFunction(reader);
}

void TFAbstractModifier::resizeEvent(QResizeEvent* e){

	painter_->setArea(rect());

	inputArea_ = painter_->getInputArea();

	computeInput_();
	update();
}

void TFAbstractModifier::paintEvent(QPaintEvent*){

	QPainter drawer(this);
	drawer.drawPixmap(rect(), painter_->getView(workCopy_));
}

} // namespace GUI
} // namespace M4D
