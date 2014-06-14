#include "MedV4D/GUI/TF/AbstractModifier.h"

namespace M4D {
namespace GUI {

AbstractModifier::AbstractModifier(TransferFunctionInterface::Ptr function, AbstractPainter::Ptr painter):
	painter_(painter),
	workCopy_(WorkCopy::Ptr(new WorkCopy(function))),
	coords_(function->getDimension()),
	ignorePoint_(-1, -1),
	toolsWidget_(NULL),
	changed_(true){
}

QWidget* AbstractModifier::getTools()
{
	if(!toolsWidget_) {
		createTools_();
	}
	return toolsWidget_;
}

TF::Size AbstractModifier::getDimension()
{
	return workCopy_->getDimension();
}

TransferFunctionInterface::Const AbstractModifier::getFunction()
{
	return workCopy_->getFunction();
}

void AbstractModifier::setHistogram(const TF::HistogramInterface::Ptr histogram){
	if(histogram->getDimension() != workCopy_->getDimension()) {
		workCopy_->setHistogram(TF::HistogramInterface::Ptr());
	} else {
		workCopy_->setHistogram(histogram);
	}
}

bool AbstractModifier::changed()
{
	if(changed_) {
		changed_ = false;
		return true;
	}
	return false;
}

M4D::Common::TimeStamp AbstractModifier::getTimeStamp()
{
	return stamp_;
}

void AbstractModifier::save(TF::XmlWriterInterface* writer)
{
	painter_->save(writer);
	workCopy_->save(writer);
	saveSettings_(writer);
}

void AbstractModifier::saveFunction(TF::XmlWriterInterface* writer)
{
	workCopy_->saveFunction(writer);
}

bool AbstractModifier::load(TF::XmlReaderInterface* reader, bool& sideError)
{
	#ifndef TF_NDEBUG
		std::cout << "Loading modifier..." << std::endl;
	#endif

	bool painterOk = painter_->load(reader);
	bool workCopyOk = workCopy_->load(reader, sideError);

	bool error = !loadSettings_(reader);
	sideError = sideError || error;

	return painterOk && workCopyOk;
}

bool AbstractModifier::loadFunction(TF::XmlReaderInterface* reader)
{
	return workCopy_->loadFunction(reader);
}

void AbstractModifier::resizeEvent(QResizeEvent* e)
{
	painter_->setArea(rect());

	inputArea_ = painter_->getInputArea();

	computeInput_();
	update();
}

void AbstractModifier::paintEvent(QPaintEvent*)
{
	QPainter drawer(this);
	drawer.drawPixmap(rect(), painter_->getView(workCopy_));
}

} // namespace GUI
} // namespace M4D
