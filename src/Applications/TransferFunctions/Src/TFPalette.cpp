#include "TFPalette.h"

namespace M4D {
namespace GUI {

TFPalette::TFPalette(QMainWindow* parent):
	QMainWindow(parent),
	ui_(new Ui::TFPalette),
	mainWindow_(parent),
	activeEditor_(emptyPalette),
	activeChanged_(false),
	creator_(parent, this){

    ui_->setupUi(this);
	setWindowTitle("Transfer Functions Palette");

	layout_ = new QVBoxLayout;
	layout_->setAlignment(Qt::AlignLeft | Qt::AlignTop);
	ui_->scrollAreaWidget->setLayout(layout_);

	ui_->removeButton->setEnabled(false);
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

	bool findNewActive = (palette_.find(activeEditor_)->second.holder->getDimension() != dataStructure_.size());
	HolderMapIt beginPalette = palette_.begin();
	HolderMapIt endPalette = palette_.end();
	for(HolderMapIt it = beginPalette; it != endPalette; ++it)
	{
		if(it->second.holder->getDimension() != dataStructure_.size())
		{
			it->second.holder->setAvailable(false);
		}
		else
		{
			it->second.holder->setDataStructure(dataStructure_);
			it->second.holder->setAvailable(true);
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

	editor->second.button->setPreview(preview);
}

TFFunctionInterface::Const TFPalette::getTransferFunction(){

	if(activeEditor_ == emptyPalette || activeEditor_ == noFunctionAvailable)
	{
		return TFFunctionInterface::Const();
	}
	
	return palette_.find(activeEditor_)->second.holder->getFunction();
}
	
TF::Size TFPalette::getDimension(){

	return dataStructure_.size();
}

TF::Size TFPalette::getDomain(const TF::Size dimension){

	if(dimension > dataStructure_.size() || dimension == 0) return 0;
	return dataStructure_[dimension-1];
}

std::vector<TFBasicHolder*> TFPalette::getEditors(){

	std::vector<TFBasicHolder*> editors;

	HolderMapIt beginPalette = palette_.begin();
	HolderMapIt endPalette = palette_.end();
	for(HolderMapIt it = beginPalette; it != endPalette; ++it)
	{
		editors.push_back(it->second.holder);
	}
	return editors;
}

void TFPalette::setHistogram(const TF::Histogram::Ptr histogram){

	histogram_ = histogram;

	HolderMapIt beginPalette = palette_.begin();
	HolderMapIt endPalette = palette_.end();
	for(HolderMapIt it = beginPalette; it != endPalette; ++it)
	{
		it->second.holder->setHistogram(histogram_);
	}
}

bool TFPalette::changed(){

	if(activeEditor_ == emptyPalette ||
		activeEditor_ == noFunctionAvailable ||
		activeChanged_ ||
		palette_.find(activeEditor_)->second.holder->changed())
	{
		activeChanged_ = false;
		return true;
	}

	return false;
}
	
M4D::Common::TimeStamp TFPalette::lastPaletteChange(){

	return lastChange_;
}

void TFPalette::addToPalette_(TFBasicHolder* holder){
	
	TF::Size addedIndex = indexer_.getIndex();

	holder->setup(mainWindow_, addedIndex);
	holder->setHistogram(histogram_);

	bool activateConnected = QObject::connect( holder, SIGNAL(Activate(TF::Size)), this, SLOT(change_activeHolder(TF::Size)));
	tfAssert(activateConnected);
	bool closeConnected = QObject::connect( holder, SIGNAL(Close(TF::Size)), this, SLOT(close_triggered(TF::Size)));
	tfAssert(closeConnected);
	
	TFPaletteButton* button = new TFPaletteButton(ui_->scrollAreaWidget, addedIndex);
	button->setup();
	layout_->addWidget(button);
	button->show();

	bool buttonConnected = QObject::connect(button, SIGNAL(Triggered(TF::Size)), this, SLOT(change_activeHolder(TF::Size)));
	tfAssert(buttonConnected);
	
	palette_.insert(std::make_pair<TF::Size, Editor>(addedIndex, Editor(holder, button)));
	
	QDockWidget* dockHolder = holder->getDockWidget();
	dockHolder->setFeatures(QDockWidget::AllDockWidgetFeatures);
	mainWindow_->addDockWidget(Qt::BottomDockWidgetArea, dockHolder);
	dockHolder->setFloating(true);
	
	bool dimMatch = (holder->getDimension() == dataStructure_.size());
	if(dimMatch) holder->setDataStructure(dataStructure_);
	if(dataStructure_.size() == 0 || dimMatch)
	{
		holder->setAvailable(true);
		change_activeHolder(addedIndex);
	}

	ui_->removeButton->setEnabled(true);
	++lastChange_;
}

void TFPalette::removeFromPalette_(const TF::Size index){

	HolderMapIt toRemoveIt = palette_.find(index);
	if(index == activeEditor_) activateNext_(toRemoveIt);

	mainWindow_->removeDockWidget(toRemoveIt->second.holder->getDockWidget());
	delete toRemoveIt->second.holder;

	layout_->removeWidget(toRemoveIt->second.button);
	delete toRemoveIt->second.button;

	palette_.erase(toRemoveIt);
	indexer_.releaseIndex(index);

	if(palette_.empty()) ui_->removeButton->setEnabled(false);
	++lastChange_;
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
			if(next != it && next->second.holder->getDimension() == dataStructure_.size()) break;
		}

		if(next != endPalette) change_activeHolder(next->second.holder->getIndex());
		else activeEditor_ = noFunctionAvailable;
	}
}

void TFPalette::resizeEvent(QResizeEvent* e){

	ui_->paletteLayoutWidget->setGeometry(ui_->paletteArea->geometry());
}

void TFPalette::close_triggered(TF::Size index){

	removeFromPalette_(index);
}

void TFPalette::change_activeHolder(TF::Size index){

	if(index == activeEditor_) return;

	Editor active;
	if(activeEditor_ >= 0)
	{
		active = palette_.find(activeEditor_)->second;
		active.button->setActive(false);
		active.holder->setActive(false);
	}

	activeEditor_ = index;
	active = palette_.find(activeEditor_)->second;
	active.button->setActive(true);
	active.holder->setActive(true);

	activeChanged_ = true;
}

void TFPalette::on_addButton_clicked(){

	TFBasicHolder* created = creator_.createTransferFunction();

	if(!created) return;
	
	addToPalette_(created);
}

void TFPalette::on_removeButton_clicked(){

	palette_.find(activeEditor_)->second.holder->close();
}

void TFPalette::closeEvent(QCloseEvent *e){
	
	while(!palette_.empty())
	{
		if(!palette_.begin()->second.holder->close()) break;
	}
	if(palette_.empty()) e->accept();
	else e->ignore();
}

} // namespace GUI
} // namespace M4D
