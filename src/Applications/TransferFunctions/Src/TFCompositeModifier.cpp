#include "TFCompositeModifier.h"

#include <TFPalette.h>
#include <TFBasicHolder.h>

#include <QtGui/QMessageBox>

namespace M4D {
namespace GUI {

TFCompositeModifier::TFCompositeModifier(
		TFAbstractFunction<TF_DIMENSION_1>::Ptr function,
		TFSimplePainter::Ptr painter,		
		TFPalette* palette):
	TFViewModifier(function, painter),
	compositeTools_(new Ui::TFCompositeModifier),
	compositeWidget_(new QWidget),
	layout_(new QVBoxLayout),
	pushUpSpacer_(new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding)),
	function_(function),
	palette_(palette),
	manager_(palette){

	compositeTools_->setupUi(compositeWidget_);

	compositeTools_->manageButton->setEnabled(manager_.refreshSelection());

	layout_->setContentsMargins(10,10,10,10);
	compositeTools_->scrollArea->setLayout(layout_);

	bool manageConnected = QObject::connect(compositeTools_->manageButton, SIGNAL(clicked()),
		this, SLOT(manageComposition_clicked()));
	tfAssert(manageConnected);

	bool delaySpinConnected = QObject::connect(compositeTools_->delaySpin, SIGNAL(valueChanged(int)),
		this, SLOT(changeChecker_intervalChange(int)));
	tfAssert(delaySpinConnected);

	bool timerConnected = QObject::connect(&changeChecker_, SIGNAL(timeout()), this, SLOT(change_check()));
	tfAssert(timerConnected);
	changeChecker_.setInterval(compositeTools_->delaySpin->value());
	changeChecker_.start();
}

TFCompositeModifier::~TFCompositeModifier(){}

void TFCompositeModifier::createTools_(){

    QFrame* separator = new QFrame();
    separator->setFrameShape(QFrame::HLine);
    separator->setFrameShadow(QFrame::Sunken);

	QVBoxLayout* layout = new QVBoxLayout();
	layout->addItem(centerWidget_(compositeWidget_));
	layout->addWidget(separator);
	layout->addItem(centerWidget_(viewWidget_));

	toolsWidget_ = new QWidget();
	toolsWidget_->setLayout(layout);
}

void TFCompositeModifier::computeInput_(){

	workCopy_->resize(1, inputArea_.width());
	workCopy_->resizeHistogram(inputArea_.width());
}

std::vector<int> TFCompositeModifier::computeZoomMoveIncrements_(const int moveX, const int moveY){

	workCopy_->moveHistogram(moveX);
	return std::vector<int>(1, moveX);
}

void TFCompositeModifier::clearLayout_(){

	layout_->removeItem(pushUpSpacer_);
	QLayoutItem* layoutIt;
	while(!layout_->isEmpty())
	{
		layoutIt = layout_->itemAt(0);
		layout_->removeItem(layoutIt);
		layoutIt->widget()->hide();
		delete layoutIt;
	}
	nameList_.clear();
}

void TFCompositeModifier::wheelEvent(QWheelEvent* e){
	
	int steps = e->delta() / 120;
	if(steps == 0) return;

	TF::PaintingPoint relativePoint = getRelativePoint_(e->x(), e->y());
	if(relativePoint == ignorePoint_) return;

	if(!altPressed_)
	{
		workCopy_->zoomHistogram(relativePoint.x, steps);
		update();
	}

	TFViewModifier::wheelEvent(e);
}

void TFCompositeModifier::changeChecker_intervalChange(int value){

	changeChecker_.setInterval(value);
}

void TFCompositeModifier::manageComposition_clicked(){

	if(manager_.refreshSelection()) manager_.exec();
	else
	{
		QMessageBox::warning(this,
			QObject::tr("Transfer Functions"),
			QObject::tr("No function available for composition.")
		);
		return;
	}
	change_check();
}

void TFCompositeModifier::change_check(){

	compositeTools_->manageButton->setEnabled(manager_.refreshSelection());

	Composition composition = manager_.getComposition();

	bool update = false;
	if(composition != composition_)
	{
		composition_.swap(composition);
		clearLayout_();	
		QLabel* editorName;
		for(Composition::iterator it = composition_.begin(); it != composition_.end(); ++it)
		{
			editorName = new QLabel(QString::fromStdString((*it)->getName()));
			nameList_.push_back(editorName);
			layout_->addWidget(editorName);
		}
		layout_->addItem(pushUpSpacer_);
		update = true;
	}
	else
	{
		QString name;
		for(TF::Size i = 0; i < composition_.size(); ++i)
		{
			if(composition_[i]->changed()) update = true;
			name = QString::fromStdString(composition_[i]->getName());
			if(nameList_[i]->text() != name) nameList_[i]->setText(name);
		}
	}

	if(update) computeResultFunction_();
}

void TFCompositeModifier::computeResultFunction_(){

	TF::Size domain = function_->getDomain(TF_DIMENSION_1);
	for(Composition::iterator it = composition_.begin(); it != composition_.end(); ++it)
	{
		if((*it)->getFunction().getDomain(TF_DIMENSION_1) != domain) return;
	}	//check if dimension change is in process

	for(TF::Size i = 0; i < domain; ++i)
	{		
		TF::Color result;
		for(Composition::iterator it = composition_.begin(); it != composition_.end(); ++it)
		{
			result += (*it)->getFunction().getRGBfColor(TF_DIMENSION_1, i);
		}
		result /= composition_.size();
		function_->setRGBfColor(TF_DIMENSION_1, i, result);
	}
	workCopy_->forceUpdate();
	changed_ = true;
	update();
}

} // namespace GUI
} // namespace M4D
