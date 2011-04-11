#include <TFBasicHolder.h>

#include <QtGui/QResizeEvent>

namespace M4D {
namespace GUI {

TFBasicHolder::TFBasicHolder(TFAbstractPainter<1>::Ptr painter,
				   TFAbstractModifier<1>::Ptr modifier,
				   TF::Types::Structure structure):
	modifier_(modifier),
	painter_(painter),
	dockTools_(NULL){	

	structure_ = structure;	
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

	update();
}

void TFBasicHolder::setDomain(const TF::Size domain){

	modifier_->getWorkCopy()->setDomain(domain);
	update();
}

void TFBasicHolder::setup(QMainWindow* mainWindow, const int index){

	TFAbstractHolder::setup(mainWindow, index);

	name_ = TF::convert<TF::Types::Predefined, std::string>(structure_.predefined);
	
	bool rereshConnected = QObject::connect( &(*modifier_), SIGNAL(RefreshView()), this, SLOT(refresh_view()));
	tfAssert(rereshConnected);
	
	QWidget* tools = modifier_->getTools();
	if(tools)
	{
		dockTools_ = new QDockWidget(this);	
		dockTools_->setWidget(tools);
		dockTools_->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);	
		dockTools_->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);	
		dockTools_->setWindowTitle(QString::fromStdString(name_ + " Tools"));
		holderMain_->addDockWidget(Qt::LeftDockWidgetArea, dockTools_);	
	}

	holderDock_->setWindowTitle(QString::fromStdString(name_));
	modifier_->setParent(this);
	modifier_->show();
}

void TFBasicHolder::resizeEvent(QResizeEvent *e){

	ui_->holderWidget->setGeometry(rect());
			
	modifier_->setGeometry(
		painterLeftTopMargin_.x,
		painterLeftTopMargin_.y, 
		width() - painterLeftTopMargin_.x - painterRightBottomMargin_.x,
		height() - painterLeftTopMargin_.y - painterRightBottomMargin_.y);

	painter_->setArea(modifier_->geometry());	
	modifier_->setInputArea(painter_->getInputArea());
}

void TFBasicHolder::paintEvent(QPaintEvent *e){

	QPainter drawer(this);
	drawer.drawPixmap(modifier_->geometry(), painter_->getView(modifier_->getWorkCopy()));
}
	
void TFBasicHolder::saveData_(TFXmlWriter::Ptr writer){
		
	writer->writeStartElement("Holder");
			
		writer->writeAttribute("Name", name_);

	writer->writeEndElement();

	painter_->save(writer);
	modifier_->getWorkCopy()->save(writer);
	modifier_->save(writer);

	modifier_->getWorkCopy()->getFunction()->save(writer);
}

bool TFBasicHolder::loadData(TFXmlReader::Ptr reader, bool& sideError){	

	#ifndef TF_NDEBUG
		std::cout << "Loading data:" << std::endl;
	#endif
	
	sideError = false;
	if(reader->readElement("Holder")) name_ = reader->readAttribute("Name");
	else sideError = true;

	bool painterLoaded = painter_->load(reader);
	bool workCopyLoaded = modifier_->getWorkCopy()->load(reader);
	bool modifierLoaded = modifier_->load(reader);
	
	bool error;
	bool functionLoaded = modifier_->getWorkCopy()->getFunction()->load(reader, error);
	
	sideError = sideError || error || !painterLoaded || !modifierLoaded || !workCopyLoaded;

	if(!functionLoaded) return false;

	fileName_ = reader->fileName();
	saved_ = true;
	return true;
}

} // namespace GUI
} // namespace M4D
