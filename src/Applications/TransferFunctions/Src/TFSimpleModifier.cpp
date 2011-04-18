#include "TFSimpleModifier.h"

namespace M4D {
namespace GUI {

TFSimpleModifier::TFSimpleModifier(
		TFAbstractFunction<TF_DIMENSION_1>::Ptr function,
		TFSimplePainter::Ptr painter):
	TFViewModifier(function, painter),
	simpleTools_(new Ui::TFSimpleModifier),
	simpleWidget_(new QWidget),
	activeView_(Active1),
	inputHelper_(),
	leftMousePressed_(false){

	simpleTools_->setupUi(simpleWidget_);

	bool changeViewConnected = QObject::connect(simpleTools_->activeViewBox, SIGNAL(currentIndexChanged(int)),
		this, SLOT(activeView_changed(int)));
	tfAssert(changeViewConnected);

	std::vector<std::string> names = painter->getComponentNames();
	for(TF::Size i = 0; i < names.size(); ++i)
	{
		simpleTools_->activeViewBox->addItem(QString::fromStdString(names[i]));
	}
	firstOnly_ = (names.size() < 3);
}

TFSimpleModifier::~TFSimpleModifier(){

	delete simpleTools_;
}

void TFSimpleModifier::createTools_(){

    QFrame* separator = new QFrame();
    separator->setFrameShape(QFrame::HLine);
    separator->setFrameShadow(QFrame::Sunken);

	QVBoxLayout* layout = new QVBoxLayout();
	layout->addItem(centerWidget_(simpleWidget_));
	layout->addWidget(separator);
	layout->addItem(centerWidget_(viewWidget_));

	toolsWidget_ = new QWidget();
	toolsWidget_->setLayout(layout);
}

void TFSimpleModifier::computeInput_(){

	workCopy_->resize(1, inputArea_.width());
	workCopy_->resizeHistogram(inputArea_.width());
}

std::vector<int> TFSimpleModifier::computeZoomMoveIncrements_(const int moveX, const int moveY){

	workCopy_->moveHistogram(moveX);
	return std::vector<int>(1, moveX);
}

void TFSimpleModifier::activeView_changed(int index){

	switch(index)
	{
		case 0:
		{
			activeView_ = Active1;
			break;
		}
		case 1:
		{
			if(firstOnly_)
			{
				activeView_ = ActiveAlpha;
			}
			else
			{
				activeView_ = Active2;
			}
			break;
		}
		case 2:
		{
			activeView_ = Active3;
			break;
		}
		case 3:
		{
			activeView_ = ActiveAlpha;
			break;
		}
		default:
		{
			tfAssert(!"Bad view selected.");
			break;
		}
	}
}

void TFSimpleModifier::wheelEvent(QWheelEvent* e){
	
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

void TFSimpleModifier::mousePressEvent(QMouseEvent *e){

	TF::PaintingPoint relativePoint = getRelativePoint_(e->x(), e->y());
	if(relativePoint == ignorePoint_) return;

	if(e->button() == Qt::RightButton)
	{
		int nextIndex = (simpleTools_->activeViewBox->currentIndex()+1) % simpleTools_->activeViewBox->count();
		simpleTools_->activeViewBox->setCurrentIndex(nextIndex);
	}
	if(e->button() == Qt::LeftButton)
	{
		leftMousePressed_ = true;
		inputHelper_ = relativePoint;
	}

	TFViewModifier::mousePressEvent(e);
}

void TFSimpleModifier::mouseReleaseEvent(QMouseEvent *e){

	TF::PaintingPoint relativePoint = getRelativePoint_(e->x(), e->y(), leftMousePressed_ || zoomMovement_);
	if(relativePoint == ignorePoint_) return;

	if(leftMousePressed_) addPoint_(relativePoint.x, relativePoint.y);
	leftMousePressed_ = false;
	update();

	TFViewModifier::mouseReleaseEvent(e);
}

void TFSimpleModifier::mouseMoveEvent(QMouseEvent *e){
	
	TF::PaintingPoint relativePoint = getRelativePoint_(e->x(), e->y(), leftMousePressed_ || zoomMovement_);
	if(relativePoint == ignorePoint_) return;

	if(leftMousePressed_)
	{
		addLine_(inputHelper_, relativePoint);
		inputHelper_ = relativePoint;
		update();
	}

	TFViewModifier::mouseMoveEvent(e);
}

void TFSimpleModifier::addPoint_(const int x, const int y){

	float yValue = y/(float)inputArea_.height();
	
	switch(activeView_)
	{
		case Active1:
		{
			workCopy_->setComponent1(TF_DIMENSION_1, x, yValue);
			if(firstOnly_)
			{
				workCopy_->setComponent2(TF_DIMENSION_1, x, yValue);
				workCopy_->setComponent3(TF_DIMENSION_1, x, yValue);
			}
			break;
		}
		case Active2:
		{
			workCopy_->setComponent2(TF_DIMENSION_1, x, yValue);
			break;
		}
		case Active3:
		{
			workCopy_->setComponent3(TF_DIMENSION_1, x, yValue);
			break;
		}
		case ActiveAlpha:
		{
			workCopy_->setAlpha(TF_DIMENSION_1, x, yValue);
			break;
		}
	}
	changed_ = true;
	++stamp_;
}

void TFSimpleModifier::addLine_(TF::PaintingPoint begin, TF::PaintingPoint end){
	
	addLine_(begin.x, begin.y, end.x, end.y);
}

void TFSimpleModifier::addLine_(int x1, int y1, int x2, int y2){
	
	if(x1==x2 && y1==y2) addPoint_(x1,y1);

	int D, ax, ay, sx, sy;

	sx = x2 - x1;
	ax = abs( sx ) << 1;

	if ( sx < 0 ) sx = -1;
	else if ( sx > 0 ) sx = 1;

	sy = y2 - y1;
	ay = abs( sy ) << 1;

	if ( sy < 0 ) sy = -1;
	else if ( sy > 0 ) sy = 1;

	if ( ax > ay )                          // x coordinate is dominant
	{
		D = ay - (ax >> 1);                   // initial D
		ax = ay - ax;                         // ay = increment0; ax = increment1

		while ( x1 != x2 )
		{
			addPoint_(x1,y1);
			if ( D >= 0 )                       // lift up the Y coordinate
			{
				y1 += sy;
				D += ax;
			}
			else
			{
				D += ay;
			}
			x1 += sx;
		}
	}
	else                                    // y coordinate is dominant
	{
		D = ax - (ay >> 1);                   // initial D
		ay = ax - ay;                         // ax = increment0; ay = increment1

		while ( y1 != y2 )
		{
			addPoint_(x1,y1);
			if ( D >= 0 )                       // lift up the X coordinate
			{
				x1 += sx;
				D += ay;
			}
			else
			{
				D += ax;
			}
			y1 += sy;
		}
	}
}

} // namespace GUI
} // namespace M4D
