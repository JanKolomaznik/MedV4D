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
	painterRightBottom_(20, 10){

	basicTools_->setupUi(this);
}

TFHolder::~TFHolder(){

	if(basicTools_) delete basicTools_;
}

M4D::Common::TimeStamp TFHolder::getLastChangeTime(){

	return modifier_->getLastChangeTime();
}

void TFHolder::setUp(TFSize index){

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

void TFHolder::updateFunction_(){

	M4D::Common::TimeStamp lastChange = modifier_->getLastChangeTime();
	if(lastChange == lastChange_) return;

	lastChange_ = lastChange;
	calculate_(modifier_->getWorkCopy(), function_->getColorMap());
}

void TFHolder::updateWorkCopy_(){
	
	calculate_(function_->getColorMap(), modifier_->getWorkCopy());
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
	if(e->button() == Qt::MidButton) mb = MouseButtonMid;

	modifier_->mousePress(e->pos().x(), e->pos().y(), mb);
}

void TFHolder::mouseReleaseEvent(QMouseEvent *e){

	modifier_->mouseRelease(e->pos().x(), e->pos().y());
}

void TFHolder::mouseMoveEvent(QMouseEvent *e){
	
	modifier_->mouseMove(e->pos().x(), e->pos().y());
	repaint();
}

void TFHolder::resizeEvent(QResizeEvent *e){
	
	basicTools_->closeButton->move(
		width() - basicTools_->closeButton->width() - painterRightBottom_.x,
		basicTools_->closeButton->geometry().y() );

	basicTools_->saveButton->move(
		basicTools_->closeButton->x() - basicTools_->saveButton->width() - painterRightBottom_.x,
		basicTools_->closeButton->y() );	

	resizePainter_();
}

void TFHolder::resizePainter_(){

	updateFunction_();

	TFArea painterArea(painterLeftTop_.x,
		painterLeftTop_.y,
		width() - painterLeftTop_.x - painterRightBottom_.x,
		height() - painterLeftTop_.y - painterRightBottom_.y);

	painter_->setArea(painterArea);

	TFArea inputArea = painter_->getInputArea();
	modifier_->setInputArea(inputArea);
	
	updateWorkCopy_();
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
	
	updateFunction_();

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

	updateWorkCopy_();
	
	return true;
}

void TFHolder::calculate_(const TFColorMapPtr input, TFColorMapPtr output){

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

		TFColorMapIt outIt = output->begin();

		TFColorMap::const_iterator inBegin = input->begin();
		TFColorMap::const_iterator inEnd = input->end();

		for(TFColorMap::const_iterator it = inBegin; it != inEnd; ++it)
		{
			TFColor computedValue(0,0,0,0);
			TFSize valueCount = inOutRatio + (int)inOutCorrection;
			for(TFSize i = 0; i < valueCount; ++i)
			{
				if(it == inEnd) return;		//TODO fail

				computedValue.component1 += it->component1;
				computedValue.component2 += it->component2;
				computedValue.component3 += it->component3;
				computedValue.alpha += it->alpha;

				if(i < (valueCount-1)) ++it;
			}
			inOutCorrection -= (int)inOutCorrection;
			inOutCorrection += corrStep;

			tfAssert(outIt != output->end());

			computedValue.component1 = computedValue.component1/valueCount;
			computedValue.component2 = computedValue.component2/valueCount;
			computedValue.component3 = computedValue.component3/valueCount;
			computedValue.alpha = computedValue.alpha/valueCount;

			*outIt = computedValue;
			++outIt;
		}
	}
	else
	{
		float outInCorrection = outputSize/(float)inputSize;
		int outInRatio = (int)(outInCorrection);	//how many input values are used for computing 1 output values
		outInCorrection -= outInRatio;
		float corrStep = outInCorrection;

		TFColorMapIt outIt = output->begin();

		TFColorMap::const_iterator inBegin = input->begin();
		TFColorMap::const_iterator inEnd = input->end();

		for(TFColorMap::const_iterator it = inBegin; it != inEnd; ++it)
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
