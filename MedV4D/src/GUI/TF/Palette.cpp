#include "MedV4D/GUI/TF/Palette.h"

namespace M4D {
namespace GUI {

Palette::Palette(QMainWindow *parent, const std::vector<TF::Size>& dataStructure):
	QMainWindow(parent),
	mParentMainWindow(parent),
	mLayout(new QGridLayout()),
	mColModulator(1),
	dataStructure_(dataStructure),
	activeChanged_(false),
	mActiveEditor(emptyPalette),
	mEditorCreator(parent, this, dataStructure),
	previewEnabled_(true)
{
	setupUi(this);
	setWindowTitle(tr("Transfer Functions Palette"));

	mLayout->setContentsMargins(10,10,10,10);
	mLayout->setAlignment(Qt::AlignCenter);
	mLayout->setSpacing(5);
	scrollAreaWidget->setLayout(mLayout);

	bool timerConnected = QObject::connect(&previewUpdater_, SIGNAL(timeout()), this, SLOT(update_previews()));
	tfAssert(timerConnected);

	previewUpdater_.setInterval(500);
	previewUpdater_.start();

	QObject::connect(&mChangeDetectionTimer, SIGNAL(timeout()), this, SLOT(detectChanges()));
	mChangeDetectionTimer.setInterval(550);
	mChangeDetectionTimer.start();

	QObject::connect(addButton, &QPushButton::clicked, this, &Palette::onAddButtonClicked);
}

Palette::~Palette()
{
}

void Palette::setupDefault()
{
	//default palette functions
}

void Palette::setDataStructure(const std::vector<TF::Size>& dataStructure)
{
	dataStructure_ = dataStructure;
	mEditorCreator.setDataStructure(dataStructure_);

	if (mActiveEditor == emptyPalette) {
		return;
	}

	bool findNewActive = (mPalette.find(mActiveEditor)->second->editor->getDimension() != dataStructure_.size());
	HolderMapIt beginPalette = mPalette.begin();
	HolderMapIt endPalette = mPalette.end();
	for(HolderMapIt it = beginPalette; it != endPalette; ++it) {
		if(it->second->editor->getDimension() != dataStructure_.size()) {
			it->second->editor->setAvailable(false);
		} else {
			it->second->editor->setDataStructure(dataStructure_);
			it->second->editor->setAvailable(true);
			it->second->updatePreview();
			if(findNewActive) {
				mActiveEditor = it->first;
			}
		}
	}
	if(findNewActive) {
		mActiveEditor = noFunctionAvailable;
	}
}

void Palette::setPreview(const QImage& preview, const int index)
{
	TF::Size editorIndex = mActiveEditor;
	if(index >= 0) editorIndex = index;

	HolderMapIt editor = mPalette.find(editorIndex);
	if(editor == mPalette.end()) return;

	editor->second->button->setPreview(preview);
	editor->second->previewUpdate = editor->second->editor->lastChange();
}

QImage
Palette::getPreview(const int index)
{

	TF::Size editorIndex = mActiveEditor;
	if(index >= 0) editorIndex = index;

	HolderMapIt editor = mPalette.find(editorIndex);
	if(editor == mPalette.end()) return QImage(getPreviewSize(), QImage::Format_RGB16);

	return editor->second->button->getPreview();
}

QSize
Palette::getPreviewSize()
{

	return QSize(PaletteButton::previewWidth, PaletteButton::previewHeight);
}

void
Palette::update_previews()
{

	if(!previewEnabled_) return;

	bool updateRequestSent = false;
	M4D::Common::TimeStamp lastChange;
	HolderMapIt beginPalette = mPalette.begin();
	HolderMapIt endPalette = mPalette.end();
	for(HolderMapIt it = beginPalette; it != endPalette; ++it)
	{
		it->second->button->setName(it->second->editor->getName());
		lastChange = it->second->editor->lastChange();
		if(!updateRequestSent && (it->first != mActiveEditor) && (it->second->previewUpdate != lastChange))
		{
			emit UpdatePreview(it->first);
			updateRequestSent = true;
		}
	}
}

void
Palette::detectChanges()
{
	HolderMapIt beginPalette = mPalette.begin();
	HolderMapIt endPalette = mPalette.end();
	for(HolderMapIt it = beginPalette; it != endPalette; ++it)
	{
		M4D::Common::TimeStamp lastChange(it->second->editor->lastChange());
		if( lastChange != it->second->lastDetectedChange ) {
			it->second->lastDetectedChange = lastChange;
			emit transferFunctionModified( it->second->editor->getIndex() );
		}
	}
}

TransferFunctionInterface::Const
Palette::getTransferFunction(const int index)
{

	TF::Size editorIndex = mActiveEditor;
	if(index >= 0) editorIndex = index;

	if(mActiveEditor == emptyPalette) {
		onAddButtonClicked();
		editorIndex = mActiveEditor;
	}

	if(mActiveEditor == emptyPalette ||
		mActiveEditor == noFunctionAvailable)
	{
		return TransferFunctionInterface::Const();
	}

	HolderMapIt editor = mPalette.find(editorIndex);
	return mPalette.find(editorIndex)->second->editor->getFunction();
}

TF::Size
Palette::getDimension()
{

	return dataStructure_.size();
}

TF::Size
Palette::getDomain(const TF::Size dimension)
{

	if(dimension > dataStructure_.size() || dimension == 0) return 0;
	return dataStructure_[dimension-1];
}

Palette::Editors
Palette::getEditors()
{

	Editors editors;

	HolderMapIt beginPalette = mPalette.begin();
	HolderMapIt endPalette = mPalette.end();
	for(HolderMapIt it = beginPalette; it != endPalette; ++it)
	{
		editors.insert(std::make_pair(it->first, it->second->editor));
	}
	return editors;
}

bool
Palette::setHistogram(const TF::HistogramInterface::Ptr histogram)
{

	TF::Size histDim = histogram->getDimension();
	if(histDim != dataStructure_.size()) return false;
	for(TF::Size i = 1; i <= histDim; ++i)
	{
		if(histogram->getDomain(i) != dataStructure_[i-1]) return false;
	}

	histogram_ = histogram;

	HolderMapIt beginPalette = mPalette.begin();
	HolderMapIt endPalette = mPalette.end();
	for(HolderMapIt it = beginPalette; it != endPalette; ++it)
	{
		it->second->editor->setHistogram(histogram_);
	}
	return true;
}

M4D::Common::TimeStamp
Palette::lastChange()
{
	ASSERT( false ) //TODO Modify for different usage
	if(mActiveEditor == emptyPalette || mActiveEditor == noFunctionAvailable)
	{
		if(activeChanged_) ++lastChange_;
		activeChanged_ = false;
	}
	else
	{
		HolderMapIt active = mPalette.find(mActiveEditor);
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
Palette::lastPaletteChange()
{

	return lastPaletteChange_;
}

void
Palette::addToPalette(Editor* editor, bool visible )
{
	bool oldBlock = blockSignals( true );
	TF::Size addedIndex = idGenerator_.NewID();

	editor->setup(mParentMainWindow, addedIndex);
	editor->setHistogram(histogram_);

	bool activateConnected = QObject::connect(editor, SIGNAL(Activate(TF::Size)), this, SLOT(change_activeHolder(TF::Size)));
	tfAssert(activateConnected);
	bool closeConnected = QObject::connect(editor, SIGNAL(Close(TF::Size)), this, SLOT(close_triggered(TF::Size)));
	tfAssert(closeConnected);

	PaletteButton* button = new PaletteButton(addedIndex);
	button->setup(editor->getName(), previewEnabled_);
	mLayout->addWidget(button);
	button->show();

	bool buttonConnected = QObject::connect(button, SIGNAL(Triggered(TF::Size)), this, SLOT(change_activeHolder(TF::Size)));
	tfAssert(buttonConnected);

	mPalette.insert(std::make_pair(addedIndex, new EditorInstance(editor, button)));

	QDockWidget* dockHolder = editor->getDockWidget();
	dockHolder->setFeatures(QDockWidget::AllDockWidgetFeatures);
	dockHolder->setAllowedAreas(Qt::AllDockWidgetAreas);
	mParentMainWindow->addDockWidget(Qt::BottomDockWidgetArea, dockHolder);
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

	reformLayout(true);

	++lastPaletteChange_;

	blockSignals( oldBlock );
	emit transferFunctionAdded( addedIndex );
	emit changedTransferFunctionSelection( addedIndex );
}

void
Palette::removeFromPalette(const TF::Size index)
{
	HolderMapIt toRemoveIt = mPalette.find(index);
	if(index == mActiveEditor) activateNext(toRemoveIt);

	mParentMainWindow->removeDockWidget(toRemoveIt->second->editor->getDockWidget());
	mLayout->removeWidget(toRemoveIt->second->button);

	delete toRemoveIt->second->button;
	//delete toRemoveIt->second->editor->close();
	delete toRemoveIt->second;
	mPalette.erase(toRemoveIt);

	reformLayout(true);

	++lastPaletteChange_;
}

void
Palette::activateNext(HolderMapIt it)
{
	if(mPalette.size() == 1) {
		mActiveEditor = emptyPalette;
	} else {
		HolderMapIt beginPalette = mPalette.begin();
		HolderMapIt endPalette = mPalette.end();

		HolderMapIt next;
		for(next = beginPalette; next != endPalette; ++next)
		{
			if(next != it && next->second->editor->getDimension() == dataStructure_.size()) break;
		}

		if(next != endPalette) change_activeHolder(next->second->editor->getIndex());
		else mActiveEditor = noFunctionAvailable;
	}
	activeChanged_ = true;
}

void Palette::resizeEvent(QResizeEvent* e){

	paletteLayoutWidget->resize(size());
	reformLayout();
}

void Palette::reformLayout(bool forceReform){

	if(mPalette.empty()) {
		return;
	}

	int newColModulator = (scrollArea->width() - 25)/(mPalette.begin()->second->button->width() + 5);
	if(newColModulator == 0) {
		newColModulator = 1;
	}

	if(!forceReform && mColModulator == newColModulator) {
		return;
	}

	mColModulator = newColModulator;

	QLayoutItem* layoutIt;
	while(!mLayout->isEmpty()) {
		layoutIt = mLayout->itemAt(0);
		mLayout->removeItem(layoutIt);
		if (layoutIt->widget()) {
			layoutIt->widget()->hide();
		}
	}

	int rowCounter = 0, colCounter = 0;
	for (auto &item : mPalette) {
		mLayout->addWidget(item.second->button, rowCounter, colCounter, Qt::AlignCenter);
		item.second->button->show();
		++colCounter;
		if(colCounter == mColModulator) {
			colCounter = 0;
			++rowCounter;
		}
	}
}

void Palette::close_triggered(TF::Size index)
{

	removeFromPalette(index);
}

void Palette::change_activeHolder(TF::Size index)
{
	LOG( "index " << index << "; activeEditor " << mActiveEditor );
	if(index == mActiveEditor) return;

	EditorInstance* active;
	HolderMapIt it;
	if(mActiveEditor >= 0) {
		it = mPalette.find(mActiveEditor);
		if ( it == mPalette.end() ) {
			D_PRINT( "Couldn't find active TF editor - id = " << mActiveEditor );
			return;
		}
		active = it->second;
		active->button->setActive(false);
		active->editor->setActive(false);
		mActiveEditor = -1;
	}

	it = mPalette.find(index);
	if ( it == mPalette.end() ) {
		D_PRINT( "Couldn't select new active TF editor - id = " << mActiveEditor );
		return;
	}
	mActiveEditor = index;
	active = it->second;
	active->button->setActive(true);
	active->editor->setActive(true);

	activeChanged_ = true;

	emit changedTransferFunctionSelection( index );
}

void Palette::onAddButtonClicked(){

	Editor* created = mEditorCreator.createEditor();

	if(!created) return;

	addToPalette(created);
}

void Palette::on_previewsCheck_toggled(bool enable){
	previewEnabled_ = enable;

	for (auto &item : mPalette) {
		item.second->button->togglePreview(previewEnabled_);
		item.second->updatePreview();
	}
}

void Palette::closeEvent(QCloseEvent *e){

	while(!mPalette.empty())
	{
		if(!mPalette.begin()->second->editor->close()) break;
	}
	if(mPalette.empty()) e->accept();
	else e->ignore();
}

} // namespace GUI
} // namespace M4D
