#include <TFHolder.h>

#include <QtGui/QResizeEvent>

namespace M4D {
namespace GUI {

TFHolder::TFHolder(QMainWindow* mainWindow,
				   TFAbstractPainter::Ptr painter,
				   TFAbstractModifier::Ptr modifier,
				   std::string title):
	holder_(new QMainWindow((QWidget*)mainWindow)),
	ui_(new Ui::TFHolder),
	modifier_(modifier),
	painter_(painter),
	button_(NULL),
	blank_(false),
	active_(false),
	index_(0),
	dockHolder_(NULL),
	dockTools_(NULL),
	painterLeftTopMargin_(20, 40),
	painterRightBottomMargin_(20, 10),
	title_(title){

	ui_->setupUi(this);
	holder_->setCentralWidget(this);
	
	lastChange_ = modifier_->getLastChangeTime();
	
	bool rereshConnected = QObject::connect( &(*modifier_), SIGNAL(RefreshView()), this, SLOT(refresh_view()));
	tfAssert(rereshConnected);
	
	QWidget* tools = modifier_->getTools();
	if(tools)
	{
		dockTools_ = new QDockWidget(this);	
		dockTools_->setWidget(tools);
		dockTools_->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);	
		dockTools_->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);	
		holder_->addDockWidget(Qt::LeftDockWidgetArea, dockTools_);	
	}
}

TFHolder::TFHolder(QMainWindow* mainWindow):
	holder_(new QMainWindow((QWidget*)mainWindow)),
	ui_(new Ui::TFHolder),
	button_(NULL),
	blank_(true),
	active_(false),
	index_(0),
	dockHolder_(NULL),
	dockTools_(NULL),
	painterLeftTopMargin_(20, 40),
	painterRightBottomMargin_(20, 10){

	ui_->setupUi(this);
	holder_->setCentralWidget(this);
}

TFHolder::~TFHolder(){

	//if(ui_) delete ui_;
}

M4D::Common::TimeStamp TFHolder::getLastChangeTime(){

	return modifier_->getLastChangeTime();
}

void TFHolder::setHistogram(TF::Histogram::Ptr histogram){

	if(!histogram) return;

	modifier_->getWorkCopy()->setHistogram(histogram);

	repaint();
}

void TFHolder::setDomain(const TF::Size domain){

	modifier_->getWorkCopy()->setDomain(domain);
	repaint();
}

void TFHolder::setup(const TF::Size index){

	index_ = index;

	title_ = title_ + " #" + TF::convert<TF::Size, std::string>(index_ + 1);
	if(dockHolder_) dockHolder_->setWindowTitle(QString::fromStdString(title_));
	if(dockTools_) dockTools_->setWindowTitle(QString::fromStdString(title_ + " Tools"));

	show();
}

bool TFHolder::connectToTFPalette(QObject *tfPalette){
		
	bool activateConnected = QObject::connect( this, SIGNAL(Activate(TF::Size)), tfPalette, SLOT(change_activeHolder(TF::Size)));
	tfAssert(activateConnected);

	bool closeConnected = QObject::connect( this, SIGNAL(Close(TF::Size)), tfPalette, SLOT(close_triggered(TF::Size)));
	tfAssert(closeConnected);

	return activateConnected &&	closeConnected;
}

bool TFHolder::createPaletteButton(QWidget *parent){

	button_ = new TFPaletteButton(parent, index_);
	button_->setup();

	bool buttonConnected = QObject::connect( button_, SIGNAL(Triggered()), this, SLOT(on_activateButton_clicked()));
	tfAssert(buttonConnected);

	return buttonConnected;
}

void TFHolder::createDockWidget(QWidget *parent){

	dockHolder_ = new QDockWidget(QString::fromStdString(title_), holder_->parentWidget());	
	dockHolder_->setWidget(holder_);
	dockHolder_->setFeatures(QDockWidget::AllDockWidgetFeatures);
}

TFPaletteButton* TFHolder::getButton() const{

	return button_;
}

QDockWidget* TFHolder::getDockWidget() const{

	return dockHolder_;
}

TF::Size TFHolder::getIndex(){

	return index_;
}

void TFHolder::activate(){

	ui_->activateButton->setChecked(true);
	button_->activate();
	active_ = true;
}

void TFHolder::deactivate(){

	ui_->activateButton->setChecked(false);
	button_->deactivate();
	active_ = false;
}

void TFHolder::paintEvent(QPaintEvent *e){

	if(blank_) return;
	QPainter drawer(this);
	drawer.drawPixmap(painterLeftTopMargin_.x, painterLeftTopMargin_.y,
		painter_->getView(modifier_->getWorkCopy()));
}

void TFHolder::mousePressEvent(QMouseEvent *e){

	if(blank_) return;
	M4D::Common::TimeStamp lastChange = modifier_->getLastChangeTime();
	if(lastChange != lastChange_)
	{
		lastChange_ = lastChange;
	}

	modifier_->mousePress(e->x(), e->y(), e->button());
}

void TFHolder::mouseReleaseEvent(QMouseEvent *e){

	if(blank_) return;
	M4D::Common::TimeStamp lastChange = modifier_->getLastChangeTime();
	if(lastChange != lastChange_)
	{
		lastChange_ = lastChange;
	}

	modifier_->mouseRelease(e->x(), e->y());
}

void TFHolder::mouseMoveEvent(QMouseEvent *e){

	if(blank_) return;
	M4D::Common::TimeStamp lastChange = modifier_->getLastChangeTime();
	if(lastChange != lastChange_)
	{
		lastChange_ = lastChange;
	}
	
	modifier_->mouseMove(e->x(), e->y());
}

void TFHolder::wheelEvent(QWheelEvent *e){

	if(blank_) return;
	int numSteps = e->delta() / 120;
	if(numSteps == 0) return;

	M4D::Common::TimeStamp lastChange = modifier_->getLastChangeTime();
	if(lastChange != lastChange_)
	{
		lastChange_ = lastChange;
	}

	modifier_->mouseWheel(numSteps, e->x(), e->y());
}

void TFHolder::resizeEvent(QResizeEvent *e){

	if(blank_) return;
	M4D::Common::TimeStamp lastChange = modifier_->getLastChangeTime();
	if(lastChange != lastChange_)
	{
		lastChange_ = lastChange;
	}

	resizePainter_();
}

void TFHolder::resizePainter_(){

	QRect painterArea(
		painterLeftTopMargin_.x,
		painterLeftTopMargin_.y,
		width() - painterLeftTopMargin_.x - painterRightBottomMargin_.x,
		height() - painterLeftTopMargin_.y - painterRightBottomMargin_.y);	
			
	painter_->setArea(painterArea);	
	modifier_->setInputArea(painter_->getInputArea());
}

void TFHolder::on_closeButton_clicked(){

	emit Close(index_);
}

void TFHolder::on_saveButton_clicked(){

	if(blank_) return;
	save();
}

void TFHolder::on_activateButton_clicked(){

	if(blank_) return;
	if(!active_) emit Activate(index_);
	else ui_->activateButton->setChecked(true);
}

void TFHolder::refresh_view(){

	repaint();
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
		QMessageBox::warning(this,
			tr("Transfer Functions"),
			tr("Cannot write file %1:\n%2.")
			.arg(fileName)
			.arg(file.errorString()));
		return;
	}

	save_(file);

	file.close();
}

void TFHolder::save_(QFile &file){
	/*
	 TFXmlWriter writer;
	 writer.write(&file, function_);*/
	 //writer.writeTestData(&file);	//testing
}

bool TFHolder::load(QFile &file){
	/*
	TFXmlReader reader;

	bool error = false;

	//reader.readTestData(&function_);	//testing
	reader.read(&file, function_, error);

	if (error || reader.error())
	{
		return false;
	}

	modifier_->getWorkCopy()->update(function_);
	*/
	return false;	//TODO
}

} // namespace GUI
} // namespace M4D
