#include "TFPalette.h"

namespace M4D {
namespace GUI {

TFPalette::TFPalette(QMainWindow* parent):
	QMainWindow(parent),
	ui_(new Ui::TFPalette),
	layout_(new QGridLayout()),
	mainWindow_(parent),
	activeEditor_(emptyPalette),
	colModulator_(1),
	activeChanged_(false),
	previewEnabled_(true),
	creator_(parent, this){

    ui_->setupUi(this);
	setWindowTitle("Transfer Functions Palette");

	layout_->setContentsMargins(10,10,10,10);
	layout_->setAlignment(Qt::AlignCenter);
	layout_->setSpacing(5);
	ui_->scrollAreaWidget->setLayout(layout_);

	bool timerConnected = QObject::connect(&previewUpdater_, SIGNAL(timeout()), this, SLOT(update_previews()));
	tfAssert(timerConnected);

	previewUpdater_.setInterval(500);
	previewUpdater_.start();
}

TFPalette::~TFPalette(){

	delete ui_;
}

void TFPalette::setupDefault(){
	//default palette functions
}

void TFPalette::setDataStructure(const std::vector<TF::Size>& dataStructure){

	dataStructure_ = dataStructure;
	creator_.setDataStructure(dataStructure_);

	if(activeEditor_ == emptyPalette) return;

	bool findNewActive = (palette_.find(activeEditor_)->second->holder->getDimension() != dataStructure_.size());
	HolderMapIt beginPalette = palette_.begin();
	HolderMapIt endPalette = palette_.end();
	for(HolderMapIt it = beginPalette; it != endPalette; ++it)
	{
		if(it->second->holder->getDimension() != dataStructure_.size())
		{
			it->second->holder->setAvailable(false);
		}
		else
		{
			it->second->holder->setDataStructure(dataStructure_);
			it->second->holder->setAvailable(true);
			it->second->updatePreview();
			if(findNewActive)
			{
				activeEditor_ = it->first;
			}
		}
	}
	if(findNewActive) activeEditor_ = noFunctionAvailable;
}

void TFPalette::setPreview(const QImage& preview, const int index){

	TF::Size editorIndex = activeEditor_;
	if(index >= 0) editorIndex = index;

	HolderMapIt editor = palette_.find(editorIndex);
	if(editor == palette_.end()) return;

	editor->second->button->setPreview(preview);
	editor->second->previewUpdate = editor->second->holder->lastChange();
}

QImage TFPalette::getPreview(const int index){

	TF::Size editorIndex = activeEditor_;
	if(index >= 0) editorIndex = index;

	HolderMapIt editor = palette_.find(editorIndex);
	if(editor == palette_.end()) return QImage(getPreviewSize(), QImage::Format_RGB16);

	return editor->second->button->getPreview();
}

QSize TFPalette::getPreviewSize(){

	return QSize(TFPaletteButton::previewWidth, TFPaletteButton::previewHeight);
}

void TFPalette::update_previews(){

	if(!previewEnabled_) return;

	bool updateRequestSent = false;
	M4D::Common::TimeStamp lastChange;
	HolderMapIt beginPalette = palette_.begin();
	HolderMapIt endPalette = palette_.end();
	for(HolderMapIt it = beginPalette; it != endPalette; ++it)
	{
		it->second->button->setName(it->second->holder->getName());
		lastChange = it->second->holder->lastChange();
		if(!updateRequestSent && (it->first != activeEditor_) && (it->second->previewUpdate != lastChange))
		{
			emit UpdatePreview(it->first);
			updateRequestSent = true;
		}
	}
}

TFFunctionInterface::Const TFPalette::getTransferFunction(const int index){

	TF::Size editorIndex = activeEditor_;
	if(index >= 0) editorIndex = index;

	if(activeEditor_ == emptyPalette)
	{
		on_addButton_clicked();
		editorIndex = activeEditor_;
	}

	if(activeEditor_ == emptyPalette ||
		activeEditor_ == noFunctionAvailable)
	{
		return TFFunctionInterface::Const();
	}

	HolderMapIt editor = palette_.find(editorIndex);	
	return palette_.find(editorIndex)->second->holder->getFunction();
}
	
TF::Size TFPalette::getDimension(){

	return dataStructure_.size();
}

TF::Size TFPalette::getDomain(const TF::Size dimension){

	if(dimension > dataStructure_.size() || dimension == 0) return 0;
	return dataStructure_[dimension-1];
}

TFPalette::Editors TFPalette::getEditors(){

	Editors editors;

	HolderMapIt beginPalette = palette_.begin();
	HolderMapIt endPalette = palette_.end();
	for(HolderMapIt it = beginPalette; it != endPalette; ++it)
	{
		editors.insert(std::make_pair(it->first, it->second->holder));
	}
	return editors;
}

void TFPalette::setHistogram(const TF::Histogram::Ptr histogram){

	histogram_ = histogram;

	HolderMapIt beginPalette = palette_.begin();
	HolderMapIt endPalette = palette_.end();
	for(HolderMapIt it = beginPalette; it != endPalette; ++it)
	{
		it->second->holder->setHistogram(histogram_);
	}
}

M4D::Common::TimeStamp TFPalette::lastChange(){

	if(activeEditor_ == emptyPalette ||	activeEditor_ == noFunctionAvailable)
	{
		if(activeChanged_) ++lastChange_;
		activeChanged_ = false;
	}
	else
	{
		HolderMapIt active = palette_.find(activeEditor_);
		Common::TimeStamp lastActiveChange = active->second->holder->lastChange();

		if(activeChanged_ || lastActiveChange != active->second->change)
		{
			++lastChange_;
			activeChanged_ = false;
			active->second->change = lastActiveChange;
		}
	}

	return lastChange_;
}
	
M4D::Common::TimeStamp TFPalette::lastPaletteChange(){

	return lastPaletteChange_;
}

void TFPalette::addToPalette_(TFBasicHolder* holder){
	
	TF::Size addedIndex = idGenerator_.NewID();

	holder->setup(mainWindow_, addedIndex);
	holder->setHistogram(histogram_);

	bool activateConnected = QObject::connect(holder, SIGNAL(Activate(TF::Size)), this, SLOT(change_activeHolder(TF::Size)));
	tfAssert(activateConnected);
	bool closeConnected = QObject::connect(holder, SIGNAL(Close(TF::Size)), this, SLOT(close_triggered(TF::Size)));
	tfAssert(closeConnected);
	
	TFPaletteButton* button = new TFPaletteButton(addedIndex);
	button->setup(holder->getName(), previewEnabled_);
	layout_->addWidget(button);
	button->show();

	bool buttonConnected = QObject::connect(button, SIGNAL(Triggered(TF::Size)), this, SLOT(change_activeHolder(TF::Size)));
	tfAssert(buttonConnected);
	
	palette_.insert(std::make_pair<TF::Size, Editor*>(addedIndex, new Editor(holder, button)));
	
	QDockWidget* dockHolder = holder->getDockWidget();
	dockHolder->setFeatures(QDockWidget::AllDockWidgetFeatures);
	dockHolder->setAllowedAreas(Qt::AllDockWidgetAreas);
	mainWindow_->addDockWidget(Qt::BottomDockWidgetArea, dockHolder);
	dockHolder->setFloating(true);
	
	bool dimMatch = (holder->getDimension() == dataStructure_.size());
	if(dimMatch) holder->setDataStructure(dataStructure_);
	if(dataStructure_.size() == 0 || dimMatch)
	{
		holder->setAvailable(true);
		change_activeHolder(addedIndex);
	}

	reformLayout_(true);

	++lastPaletteChange_;
}

void TFPalette::removeFromPalette_(const TF::Size index){

	HolderMapIt toRemoveIt = palette_.find(index);
	if(index == activeEditor_) activateNext_(toRemoveIt);

	mainWindow_->removeDockWidget(toRemoveIt->second->holder->getDockWidget());
	layout_->removeWidget(toRemoveIt->second->button);

	delete toRemoveIt->second->button;
	delete toRemoveIt->second->holder;
	delete toRemoveIt->second;
	palette_.erase(toRemoveIt);

	reformLayout_(true);

	++lastPaletteChange_;
}

void TFPalette::activateNext_(HolderMapIt it){
	
	if(palette_.size() == 1)
	{
		activeEditor_ = emptyPalette;
	}
	else
	{
		HolderMapIt beginPalette = palette_.begin();
		HolderMapIt endPalette = palette_.end();

		HolderMapIt next;
		for(next = beginPalette; next != endPalette; ++next)
		{
			if(next != it && next->second->holder->getDimension() == dataStructure_.size()) break;
		}

		if(next != endPalette) change_activeHolder(next->second->holder->getIndex());
		else activeEditor_ = noFunctionAvailable;
	}
	activeChanged_ = true;
}

void TFPalette::resizeEvent(QResizeEvent* e){

	ui_->paletteLayoutWidget->resize(size());
	reformLayout_();
}

void TFPalette::reformLayout_(bool forceReform){
	
	if(palette_.empty()) return;

	TF::Size newColModulator = (ui_->scrollArea->width() - 25)/(palette_.begin()->second->button->width() + 5);
	if(newColModulator == 0) newColModulator = 1;

	if(!forceReform && colModulator_ == newColModulator) return;

	colModulator_ = newColModulator;

	QLayoutItem* layoutIt;
	while(!layout_->isEmpty())
	{
		layoutIt = layout_->itemAt(0);
		layout_->removeItem(layoutIt);
		layoutIt->widget()->hide();
	}

	TF::Size rowCounter = 0, colCounter = 0;
	HolderMapIt beginPalette = palette_.begin();
	HolderMapIt endPalette = palette_.end();
	for(HolderMapIt it = beginPalette; it != endPalette; ++it)
	{
		layout_->addWidget(it->second->button, rowCounter, colCounter, Qt::AlignCenter);
		it->second->button->show();
		++colCounter;
		if(colCounter == colModulator_)
		{
			colCounter = 0;
			++rowCounter;
		}
	}
}

void TFPalette::close_triggered(TF::Size index){

	removeFromPalette_(index);
}

void TFPalette::change_activeHolder(TF::Size index){

	if(index == activeEditor_) return;

	Editor* active;
	if(activeEditor_ >= 0)
	{
		active = palette_.find(activeEditor_)->second;
		active->button->setActive(false);
		active->holder->setActive(false);
	}

	activeEditor_ = index;
	active = palette_.find(activeEditor_)->second;
	active->button->setActive(true);
	active->holder->setActive(true);

	activeChanged_ = true;
}

void TFPalette::on_addButton_clicked(){

	TFBasicHolder* created = creator_.createTransferFunction();

	if(!created) return;
	
	addToPalette_(created);
}

void TFPalette::on_previewsCheck_toggled(bool enable){

	previewEnabled_ = enable;	

	HolderMapIt beginPalette = palette_.begin();
	HolderMapIt endPalette = palette_.end();
	for(HolderMapIt it = beginPalette; it != endPalette; ++it)
	{
		it->second->button->togglePreview(previewEnabled_);
		it->second->updatePreview();
	}
}

void TFPalette::closeEvent(QCloseEvent *e){
	
	while(!palette_.empty())
	{
		if(!palette_.begin()->second->holder->close()) break;
	}
	if(palette_.empty()) e->accept();
	else e->ignore();
}

} // namespace GUI
} // namespace M4D
