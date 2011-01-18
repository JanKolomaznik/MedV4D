#include <TFHolder.h>

#include <QtGui/QResizeEvent>

namespace M4D {
namespace GUI {

TFHolder::TFHolder(QMainWindow* mainWindow,
				   TFAbstractFunction::Ptr function,
				   TFAbstractModifier::Ptr modifier,
				   TFAbstractPainter::Ptr painter,
				   TFHolder::Type type):
	QMainWindow((QWidget*)mainWindow),
	basicTools_(new Ui::TFHolder),
	function_(function),
	modifier_(modifier),
	painter_(painter),
	button_(NULL),
	type_(type),
	setup_(false),
	index_(0),
	dockWidget_(NULL),
	painterLeftTop_(20, 40),
	painterRightBottom_(20, 10),
	zoomMovement_(false){

	basicTools_->setupUi(this);
	lastChange_ = modifier_->getLastChangeTime();
}

TFHolder::~TFHolder(){

	if(basicTools_) delete basicTools_;
}

M4D::Common::TimeStamp TFHolder::getLastChangeTime(){

	return modifier_->getLastChangeTime();
}

void TFHolder::setUp(const TFSize& index){

	index_ = index;

	show();
}

bool TFHolder::connectToTFPalette(QObject *tfPalette){
		
	bool activateConnected = QObject::connect( this, SIGNAL(Activate(TFSize)), tfPalette, SLOT(change_activeHolder(TFSize)));
	tfAssert(activateConnected);

	bool closeConnected = QObject::connect( this, SIGNAL(Close(TFSize)), tfPalette, SLOT(close_triggered(TFSize)));
	tfAssert(closeConnected);

	return activateConnected &&
		closeConnected;
}

bool TFHolder::createPaletteButton(QWidget *parent){

	button_ = new TFPaletteButton(parent, index_);

	bool buttonConnected = QObject::connect( button_, SIGNAL(Triggered()), this, SLOT(on_activateButton_clicked()));
	tfAssert(buttonConnected);

	return buttonConnected;
}

void TFHolder::createDockWidget(QWidget *parent){

	QString qTitle = QString::fromStdString(convert<TFHolder::Type, std::string>(type_) +
		" #" + convert<TFSize, std::string>(index_ + 1));

	dockWidget_ = new QDockWidget(qTitle, parent);	
	dockWidget_->setWidget(this);
	dockWidget_->setFeatures(QDockWidget::AllDockWidgetFeatures);
}

TFHolder::Type TFHolder::getType() const{

	return type_;
}

TFPaletteButton* TFHolder::getButton() const{

	return button_;
}

QDockWidget* TFHolder::getDockWidget() const{

	return dockWidget_;
}

TFSize TFHolder::getIndex(){

	return index_;
}

void TFHolder::save(){

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

void TFHolder::paintEvent(QPaintEvent *e){

	QPainter drawer(this);
	painter_->drawBackground(&drawer);
	painter_->drawData(&drawer, modifier_->getWorkCopy());
	//TODO if histogram enabled
	//painter_->drawHistogram(&drawer, histogram_);
}

void TFHolder::mousePressEvent(QMouseEvent *e){

	MouseButton mb(MouseButtonLeft);
	if(e->button() == Qt::RightButton) mb = MouseButtonRight;
	if(e->button() == Qt::MidButton)
	{
		zoomMovement_ = true;
		zoomMoveHelper_.x = e->x();
		zoomMoveHelper_.y = e->y();
		mb = MouseButtonMid;
	}

	modifier_->mousePress(e->x(), e->y(), mb);
}

void TFHolder::mouseReleaseEvent(QMouseEvent *e){

	if(e->button() == Qt::MidButton) zoomMovement_ = false;

	modifier_->mouseRelease(e->x(), e->y());

	repaint();
}

void TFHolder::mouseMoveEvent(QMouseEvent *e){
	
	modifier_->mouseMove(e->x(), e->y());

	if(zoomMovement_)
	{
		M4D::Common::TimeStamp lastChange = modifier_->getLastChangeTime();
		if(lastChange != lastChange_)
		{
			modifier_->getWorkCopy()->updateFunction(function_);
			lastChange_ = lastChange;
		}

		TFSize x = e->x();
		TFSize y = e->y();
		if(x < painter_->getInputArea().x ||
			x > (painter_->getInputArea().x + painter_->getInputArea().width) ||
			y < painter_->getInputArea().y ||
			y > (painter_->getInputArea().y + painter_->getInputArea().height))
		{
			return;
		}

		modifier_->getWorkCopy()->move(zoomMoveHelper_.x - x, y - zoomMoveHelper_.y);

		zoomMoveHelper_.x = x;
		zoomMoveHelper_.y = y;

		modifier_->getWorkCopy()->update(function_);
	}
	repaint();
}

void TFHolder::resizeEvent(QResizeEvent *e){
	
	basicTools_->closeButton->move(
		width() - basicTools_->closeButton->width() - painterRightBottom_.x,
		basicTools_->closeButton->y() );

	basicTools_->saveButton->move(
		basicTools_->closeButton->x() - basicTools_->saveButton->width() - painterRightBottom_.x,
		basicTools_->closeButton->y() );	

	resizePainter_();
}

void TFHolder::resizePainter_(){

	M4D::Common::TimeStamp lastChange = modifier_->getLastChangeTime();
	if(lastChange != lastChange_)
	{
		modifier_->getWorkCopy()->updateFunction(function_);
		lastChange_ = lastChange;
	}

	TFArea painterArea(painterLeftTop_.x,
		painterLeftTop_.y,
		width() - painterLeftTop_.x - painterRightBottom_.x,
		height() - painterLeftTop_.y - painterRightBottom_.y);
	
	TFSize maxPainterWidth = (TFSize)(function_->getDomain()/modifier_->getWorkCopy()->zoom());
	if(painterArea.width > maxPainterWidth) painterArea.width = maxPainterWidth;
	
	painter_->setArea(painterArea);
	modifier_->setInputArea(painter_->getInputArea());
	
	modifier_->getWorkCopy()->update(function_);
}

void TFHolder::wheelEvent(QWheelEvent *e){

	int numSteps = e->delta() / 120;
	if(numSteps == 0) return;

	TFSize x = e->x() - painter_->getInputArea().x;
	TFSize y = painter_->getInputArea().height - (e->y() - painter_->getInputArea().y);
	if(x < 0 ||
		x > painter_->getInputArea().width ||
		y < 0 ||
		y > painter_->getInputArea().height)
	{
		return;
	}

	M4D::Common::TimeStamp lastChange = modifier_->getLastChangeTime();
	if(lastChange != lastChange_)
	{
		modifier_->getWorkCopy()->updateFunction(function_);
		lastChange_ = lastChange;
	}

	if(numSteps > 0) modifier_->getWorkCopy()->zoomIn(numSteps, x, y);
	else modifier_->getWorkCopy()->zoomOut(-numSteps, x, y);

	//modifier_->getWorkCopy()->update(function_);	//TODO resize ??
	resizePainter_();

	repaint();
}

void TFHolder::on_closeButton_clicked(){

	emit Close(index_);
}

void TFHolder::on_saveButton_clicked(){

	save();
}

void TFHolder::on_activateButton_clicked(){

	emit Activate(index_);
}

void TFHolder::save_(QFile &file){
	
	modifier_->getWorkCopy()->updateFunction(function_);

	 TFXmlWriter writer;
	 writer.write(&file, function_);
	 //writer.writeTestData(&file);	//testing
}

bool TFHolder::load_(QFile &file){
	
	TFXmlReader reader;

	bool error = false;

	//reader.readTestData(&function_);	//testing
	reader.read(&file, function_, error);

	if (error || reader.error())
	{
		return false;
	}

	modifier_->getWorkCopy()->update(function_);
	
	return true;
}

} // namespace GUI
} // namespace M4D
