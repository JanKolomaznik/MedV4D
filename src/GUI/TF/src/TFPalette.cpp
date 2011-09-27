#include "GUI/TF/TFPalette.h"

namespace M4D {
namespace GUI {

TFPalette::TFPalette(QMainWindow* parent, const std::vector<TF::Size>& dataStructure):
	QMainWindow(parent),
	ui_(new Ui::TFPalette),
	mainWindow_(parent),
	layout_(new QGridLayout()),
	colModulator_(1),
	dataStructure_(dataStructure),
	activeChanged_(false),
	activeEditor_(emptyPalette),
	creator_(parent, this, dataStructure),
	previewEnabled_(true)
{

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

	QObject::connect(&mChangeDetectionTimer, SIGNAL(timeout()), this, SLOT(detectChanges()));
	mChangeDetectionTimer.setInterval(550);
	mChangeDetectionTimer.start();
}

TFPalette::~TFPalette(){

	delete ui_;
}

void TFPalette::setupDefault()
{
	//default palette functions
}

void TFPalette::setDataStructure(const std::vector<TF::Size>& dataStructure)
{

	dataStructure_ = dataStructure;
	creator_.setDataStructure(dataStructure_);

	if(activeEditor_ == emptyPalette) return;

	bool findNewActive = (palette_.find(activeEditor_)->second->editor->getDimension() != dataStructure_.size());
	HolderMapIt beginPalette = palette_.begin();
	HolderMapIt endPalette = palette_.end();
	for(HolderMapIt it = beginPalette; it != endPalette; ++it)
	{
		if(it->second->editor->getDimension() != dataStructure_.size())
		{
			it->second->editor->setAvailable(false);
		}
		else
		{
			it->second->editor->setDataStructure(dataStructure_);
			it->second->editor->setAvailable(true);
			it->second->updatePreview();
			if(findNewActive)
			{
				activeEditor_ = it->first;
			}
		}
	}
	if(findNewActive) activeEditor_ = noFunctionAvailable;
}

void TFPalette::setPreview(const QImage& preview, const int index)
{

	TF::Size editorIndex = activeEditor_;
	if(index >= 0) editorIndex = index;

	HolderMapIt editor = palette_.find(editorIndex);
	if(editor == palette_.end()) return;

	editor->second->button->setPreview(preview);
	editor->second->previewUpdate = editor->second->editor->lastChange();
}

QImage 
TFPalette::getPreview(const int index)
{

	TF::Size editorIndex = activeEditor_;
	if(index >= 0) editorIndex = index;

	HolderMapIt editor = palette_.find(editorIndex);
	if(editor == palette_.end()) return QImage(getPreviewSize(), QImage::Format_RGB16);

	return editor->second->button->getPreview();
}

QSize 
TFPalette::getPreviewSize()
{

	return QSize(TFPaletteButton::previewWidth, TFPaletteButton::previewHeight);
}

void 
TFPalette::update_previews()
{

	if(!previewEnabled_) return;

	bool updateRequestSent = false;
	M4D::Common::TimeStamp lastChange;
	HolderMapIt beginPalette = palette_.begin();
	HolderMapIt endPalette = palette_.end();
	for(HolderMapIt it = beginPalette; it != endPalette; ++it)
	{
		it->second->button->setName(it->second->editor->getName());
		lastChange = it->second->editor->lastChange();
		if(!updateRequestSent && (it->first != activeEditor_) && (it->second->previewUpdate != lastChange))
		{
			emit UpdatePreview(it->first);
			updateRequestSent = true;
		}
	}
}

void
TFPalette::detectChanges()
{
	HolderMapIt beginPalette = palette_.begin();
	HolderMapIt endPalette = palette_.end();
	for(HolderMapIt it = beginPalette; it != endPalette; ++it)
	{
		M4D::Common::TimeStamp lastChange(it->second->editor->lastChange());
		if( lastChange != it->second->lastDetectedChange ) {
			it->second->lastDetectedChange = lastChange;
			emit transferFunctionModified( it->second->editor->getIndex() );
		}
	}
}

TFFunctionInterface::Const 
TFPalette::getTransferFunction(const int index)
{

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
	return palette_.find(editorIndex)->second->editor->getFunction();
}
	
TF::Size 
TFPalette::getDimension()
{

	return dataStructure_.size();
}

TF::Size 
TFPalette::getDomain(const TF::Size dimension)
{

	if(dimension > dataStructure_.size() || dimension == 0) return 0;
	return dataStructure_[dimension-1];
}

TFPalette::Editors& 
TFPalette::getEditors()
{

	Editors editors;

	HolderMapIt beginPalette = palette_.begin();
	HolderMapIt endPalette = palette_.end();
	for(HolderMapIt it = beginPalette; it != endPalette; ++it)
	{
		editors.insert(std::make_pair(it->first, it->second->editor));
	}
	return editors;
}

bool 
TFPalette::setHistogram(const TF::HistogramInterface::Ptr histogram)
{

	TF::Size histDim = histogram->getDimension();
	if(histDim != dataStructure_.size()) return false;
	for(TF::Size i = 1; i <= histDim; ++i)
	{
		if(histogram->getDomain(i) != dataStructure_[i-1]) return false;
	}

	histogram_ = histogram;

	HolderMapIt beginPalette = palette_.begin();
	HolderMapIt endPalette = palette_.end();
	for(HolderMapIt it = beginPalette; it != endPalette; ++it)
	{
		it->second->editor->setHistogram(histogram_);
	}
	return true;
}

M4D::Common::TimeStamp 
TFPalette::lastChange()
{
	ASSERT( false ) //TODO Modify for different usage
	if(activeEditor_ == emptyPalette || activeEditor_ == noFunctionAvailable)
	{
		if(activeChanged_) ++lastChange_;
		activeChanged_ = false;
	}
	else
	{
		HolderMapIt active = palette_.find(activeEditor_);
		Common::TimeStamp lastActiveChange = active->second->editor->lastChange();

		if(activeChanged_ || lastActiveChange != active->second->change)
		{
			++lastChange_;
			activeChanged_ = false;
			active->second->change = lastActiveChange;
		}
	}

	return lastChange_;
}
	
M4D::Common::TimeStamp 
TFPalette::lastPaletteChange()
{

	return lastPaletteChange_;
}

void 
TFPalette::addToPalette_(TFEditor* editor, bool visible )
{
	bool oldBlock = blockSignals( true );
	TF::Size addedIndex = idGenerator_.NewID();

	editor->setup(mainWindow_, addedIndex);
	editor->setHistogram(histogram_);

	bool activateConnected = QObject::connect(editor, SIGNAL(Activate(TF::Size)), this, SLOT(change_activeHolder(TF::Size)));
	tfAssert(activateConnected);
	bool closeConnected = QObject::connect(editor, SIGNAL(Close(TF::Size)), this, SLOT(close_triggered(TF::Size)));
	tfAssert(closeConnected);
	
	TFPaletteButton* button = new TFPaletteButton(addedIndex);
	button->setup(editor->getName(), previewEnabled_);
	layout_->addWidget(button);
	button->show();

	bool buttonConnected = QObject::connect(button, SIGNAL(Triggered(TF::Size)), this, SLOT(change_activeHolder(TF::Size)));
	tfAssert(buttonConnected);
	
	palette_.insert(std::make_pair<TF::Size, Editor*>(addedIndex, new Editor(editor, button)));
	
	QDockWidget* dockHolder = editor->getDockWidget();
	dockHolder->setFeatures(QDockWidget::AllDockWidgetFeatures);
	dockHolder->setAllowedAreas(Qt::AllDockWidgetAreas);
	mainWindow_->addDockWidget(Qt::BottomDockWidgetArea, dockHolder);
	dockHolder->move( 100, 100 );
	dockHolder->setVisible( visible );
	dockHolder->setFloating(true);
	
	bool dimMatch = (editor->getDimension() == dataStructure_.size());
	if(dimMatch) editor->setDataStructure(dataStructure_);
	if(dataStructure_.size() == 0 || dimMatch)
	{
		editor->setAvailable(true);
		change_activeHolder(addedIndex);
	}

	reformLayout_(true);

	++lastPaletteChange_;

	blockSignals( oldBlock );
	emit transferFunctionAdded( addedIndex );
	emit changedTransferFunctionSelection( addedIndex );
}

void 
TFPalette::removeFromPalette_(const TF::Size index)
{
	HolderMapIt toRemoveIt = palette_.find(index);
	if(index == activeEditor_) activateNext_(toRemoveIt);

	mainWindow_->removeDockWidget(toRemoveIt->second->editor->getDockWidget());
	layout_->removeWidget(toRemoveIt->second->button);

	delete toRemoveIt->second->button;
	//delete toRemoveIt->second->editor->close();
	delete toRemoveIt->second;
	palette_.erase(toRemoveIt);

	reformLayout_(true);

	++lastPaletteChange_;
}

void 
TFPalette::activateNext_(HolderMapIt it)
{
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
			if(next != it && next->second->editor->getDimension() == dataStructure_.size()) break;
		}

		if(next != endPalette) change_activeHolder(next->second->editor->getIndex());
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

void TFPalette::close_triggered(TF::Size index)
{

	removeFromPalette_(index);
}

void TFPalette::change_activeHolder(TF::Size index)
{
	LOG( "index " << index << "; activeEditor " << activeEditor_ );
	if(index == activeEditor_) return;

	Editor* active;
	HolderMapIt it;
	if(activeEditor_ >= 0)
	{
		it = palette_.find(activeEditor_);
		if ( it == palette_.end() ) {
			D_PRINT( "Couldn't find active TF editor - id = " << activeEditor_ );
			return;
		}
		active = it->second;
		active->button->setActive(false);
		active->editor->setActive(false);
		activeEditor_ = -1;
	}

	it = palette_.find(index);
	if ( it == palette_.end() ) {
		D_PRINT( "Couldn't select new active TF editor - id = " << activeEditor_ );
		return;
	}
	activeEditor_ = index;
	active = it->second;
	active->button->setActive(true);
	active->editor->setActive(true);

	activeChanged_ = true;

	emit changedTransferFunctionSelection( index );
}

void TFPalette::on_addButton_clicked(){

	TFEditor* created = creator_.createEditor();

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
		if(!palette_.begin()->second->editor->close()) break;
	}
	if(palette_.empty()) e->accept();
	else e->ignore();
}

} // namespace GUI
} // namespace M4D
