#include <TFBasicHolder.h>

#include <QtGui/QResizeEvent>

namespace M4D {
namespace GUI {

TFBasicHolder::TFBasicHolder(QMainWindow* mainWindow,
				   TFAbstractPainter<1>::Ptr painter,
				   TFAbstractModifier<1>::Ptr modifier,
				   TF::Types::Structure structure):
	holder_(new QMainWindow((QWidget*)mainWindow)),
	ui_(new Ui::TFBasicHolder),
	modifier_(modifier),
	painter_(painter),
	button_(NULL),
	active_(false),
	index_(0),
	dockHolder_(NULL),
	dockTools_(NULL),
	painterLeftTopMargin_(20, 40),
	painterRightBottomMargin_(20, 10){

	ui_->setupUi(this);
	holder_->setCentralWidget(this);

	structure_ = structure;
	title_ = TF::convert<TF::Types::Predefined, std::string>(structure_.predefined);
	
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
	
	show();
	ui_->activateButton->hide();
}

TFBasicHolder::~TFBasicHolder(){

	//if(ui_) delete ui_;
}

TFApplyFunctionInterface::Ptr TFBasicHolder::functionToApply_(){

	return modifier_->getWorkCopy()->getFunction();
}


bool TFBasicHolder::changed(){

	return modifier_->changed();
}

void TFBasicHolder::setHistogram(TF::Histogram::Ptr histogram){

	if(!histogram) return;

	modifier_->getWorkCopy()->setHistogram(histogram);

	repaint();
}

void TFBasicHolder::setDomain(const TF::Size domain){

	modifier_->getWorkCopy()->setDomain(domain);
	repaint();
}

void TFBasicHolder::setup(const TF::Size index){

	index_ = index;

	title_ = title_ + TF::convert<TF::Size, std::string>(index_ + 1);
	setWindowTitle(QString::fromStdString(title_));
	if(dockHolder_) dockHolder_->setWindowTitle(QString::fromStdString(title_));
	if(dockTools_) dockTools_->setWindowTitle(QString::fromStdString(title_ + " Tools"));
	
	ui_->activateButton->show();
}

bool TFBasicHolder::connectToTFPalette(QObject *tfPalette){
		
	bool activateConnected = QObject::connect( this, SIGNAL(Activate(TF::Size)), tfPalette, SLOT(change_activeHolder(TF::Size)));
	tfAssert(activateConnected);

	bool closeConnected = QObject::connect( this, SIGNAL(Close(TF::Size)), tfPalette, SLOT(close_triggered(TF::Size)));
	tfAssert(closeConnected);

	return activateConnected &&	closeConnected;
}

bool TFBasicHolder::createPaletteButton(QWidget *parent){

	button_ = new TFPaletteButton(parent, index_);
	button_->setup();

	bool buttonConnected = QObject::connect( button_, SIGNAL(Triggered()), this, SLOT(on_activateButton_clicked()));
	tfAssert(buttonConnected);

	return buttonConnected;
}

void TFBasicHolder::createDockWidget(QWidget *parent){

	dockHolder_ = new QDockWidget(QString::fromStdString(title_), holder_->parentWidget());	
	dockHolder_->setWidget(holder_);
	dockHolder_->setFeatures(QDockWidget::AllDockWidgetFeatures);
}

TFPaletteButton* TFBasicHolder::getButton() const{

	return button_;
}

QDockWidget* TFBasicHolder::getDockWidget() const{

	return dockHolder_;
}

TF::Size TFBasicHolder::getIndex(){

	return index_;
}

void TFBasicHolder::activate(){

	ui_->activateButton->setChecked(true);
	button_->activate();
	active_ = true;
}

void TFBasicHolder::deactivate(){

	ui_->activateButton->setChecked(false);
	button_->deactivate();
	active_ = false;
}

void TFBasicHolder::paintEvent(QPaintEvent *e){

	QPainter drawer(this);
	drawer.drawPixmap(painterLeftTopMargin_.x, painterLeftTopMargin_.y,
		painter_->getView(modifier_->getWorkCopy()));
}

void TFBasicHolder::mousePressEvent(QMouseEvent *e){

	modifier_->mousePress(e->x(), e->y(), e->button());
}

void TFBasicHolder::mouseReleaseEvent(QMouseEvent *e){

	modifier_->mouseRelease(e->x(), e->y());
}

void TFBasicHolder::mouseMoveEvent(QMouseEvent *e){

	modifier_->mouseMove(e->x(), e->y());
}

void TFBasicHolder::wheelEvent(QWheelEvent *e){

	int numSteps = e->delta() / 120;
	if(numSteps == 0) return;

	modifier_->mouseWheel(numSteps, e->x(), e->y());
}

void TFBasicHolder::resizeEvent(QResizeEvent *e){

	resizePainter_();
}

void TFBasicHolder::resizePainter_(){

	QRect painterArea(
		painterLeftTopMargin_.x,
		painterLeftTopMargin_.y,
		width() - painterLeftTopMargin_.x - painterRightBottomMargin_.x,
		height() - painterLeftTopMargin_.y - painterRightBottomMargin_.y);	
			
	painter_->setArea(painterArea);	
	modifier_->setInputArea(painter_->getInputArea());
}

void TFBasicHolder::on_closeButton_clicked(){

	emit Close(index_);
}

void TFBasicHolder::on_saveButton_clicked(){

	save();
}

void TFBasicHolder::on_activateButton_clicked(){

	if(!active_) emit Activate(index_);
	else ui_->activateButton->setChecked(true);
}

void TFBasicHolder::refresh_view(){

	repaint();
}
	
void TFBasicHolder::saveData_(TFXmlWriter::Ptr writer){
		
	painter_->save(writer);
	modifier_->save(writer);
	modifier_->getWorkCopy()->getFunction()->save(writer);
}

bool TFBasicHolder::loadData(TFXmlReader::Ptr reader, bool& sideError){	

	#ifndef TF_NDEBUG
		std::cout << "Loading data..." << std::endl;
	#endif

	sideError = false;
	bool error;

	bool painterLoaded = painter_->load(reader, error);
	sideError = sideError || error;

	bool modifierLoaded = modifier_->load(reader, error);
	sideError = sideError || error;

	bool functionLoaded = modifier_->getWorkCopy()->getFunction()->load(reader, error);
	sideError = sideError || error;

	if(painterLoaded && modifierLoaded && functionLoaded)
	{
		fileName_ = reader->fileName();
		return true;
	}
	return false;
}

} // namespace GUI
} // namespace M4D
