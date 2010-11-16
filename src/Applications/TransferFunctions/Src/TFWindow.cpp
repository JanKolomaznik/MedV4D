#include "TFWindow.h"

namespace M4D {
namespace GUI {

TFWindow::TFWindow(): ui_(new Ui::TFWindow), activeHolder_(-1){

    ui_->setupUi(this);
}

TFWindow::~TFWindow(){

	TFPaletteIt begin = palette_.begin();
	TFPaletteIt end = palette_.end();
	for(TFPaletteIt it = begin; it != end; ++it)
	{
		delete (*it);
	}
}

void TFWindow::setupDefault(){
	
	newTF_triggered(TFHOLDER_GRAYSCALE);
}

void TFWindow::addToPalette_(TFAbstractHolder* holder){
	
	bool resizeConnected = QObject::connect( this, SIGNAL(ResizeHolder(const QRect)), holder, SLOT(size_changed(const QRect&)));
	tfAssert(resizeConnected);

	TFSize addedIndex = palette_.size();

	TFPaletteButton* paletteButton = new TFPaletteButton(ui_->paletteArea, holder, addedIndex);
	paletteButton->setUpHolder(ui_->holderArea);
	paletteButton->showHolder();
	palette_.push_back(paletteButton);

	ui_->paletteLayout->addWidget(paletteButton);
	
	bool paletteConnected = QObject::connect( paletteButton, SIGNAL(TFPaletteSignal(const TFSize&)), this, SLOT(change_holder(const TFSize&)));
	tfAssert(paletteConnected);

	change_holder(addedIndex);
}

void TFWindow::removeActiveFromPalette_(){

	int paletteSize = palette_.size();
	if(paletteSize > 1)
	{		
		delete palette_[activeHolder_];
		for(int i = activeHolder_; i < (paletteSize-1); ++i)
		{
			palette_[i] = palette_[i+1];
			palette_[i]->changeIndex(i);
		}
		palette_.pop_back();

		int newPaletteSize = palette_.size();
		if(activeHolder_ >= newPaletteSize) --activeHolder_;
		change_holder(activeHolder_, true);
	}
	else
	{
		exit(0);
	}
}

void TFWindow::createMenu(QMenuBar* menubar){

	menuTF_ = new QMenu(QString::fromStdString("Transfer Function"), menubar);

	//menu new
	menuNew_ = new QMenu(QString::fromStdString("New"), menuTF_);	
	tfActions_ = TFHolderFactory::createMenuTFActions(menuNew_);

	TFActionsIt begin = tfActions_.begin();
	TFActionsIt end = tfActions_.end();
	for(TFActionsIt it = begin; it!=end; ++it)
	{
		bool menuActionConnected = QObject::connect( *it, SIGNAL(TFActionClicked(const TFHolderType&)), this, SLOT(newTF_triggered(const TFHolderType&)));
		tfAssert(menuActionConnected);
		menuNew_->addAction(*it);
	}

	menuTF_->addAction(menuNew_->menuAction());

	//action load
	actionLoad_ = new QAction(QString::fromStdString("Load"), menuTF_);
	bool loadConnected = QObject::connect( actionLoad_, SIGNAL(triggered()), this, SLOT(load_triggered()));
	tfAssert(loadConnected);
	actionLoad_->setShortcut(QKeySequence(QObject::tr("Ctrl+L")));
	menuTF_->addAction(actionLoad_);

	//action save
	actionSave_ = new QAction(QString::fromStdString("Save"), menuTF_);
	actionSave_->setEnabled(false);
	bool saveConnected = QObject::connect( actionSave_, SIGNAL(triggered()), this, SLOT(save_triggered()));
	tfAssert(saveConnected);
	actionSave_->setShortcut(QKeySequence::Save);
	menuTF_->addAction(actionSave_);

	//separator
	menuTF_->addSeparator();	

	//action exit
	actionExit_ = new QAction(QString::fromStdString("Close"), menuTF_);
	bool exitConnected = QObject::connect( actionExit_, SIGNAL(triggered()), this, SLOT(close_triggered()));
	tfAssert(exitConnected);
	actionExit_->setShortcut(QKeySequence::Close);
	menuTF_->addAction(actionExit_);
	
	menuTF_->setEnabled(true);
	menubar->addAction(menuTF_->menuAction());
}

void TFWindow::resizeEvent(QResizeEvent* e){

	ui_->holderArea->resize(width(), (3*height())/4); //holder zabira 3/4
	ui_->paletteArea->resize(width(), height() - ui_->holderArea->height());	//paleta zabira zbytek
	ui_->paletteArea->move(0, ui_->holderArea->height());

	emit ResizeHolder(ui_->holderArea->rect());
}

void TFWindow::close_triggered(){

	removeActiveFromPalette_();
}

void TFWindow::save_triggered(){

	palette_[activeHolder_]->saveHolder();
}

void TFWindow::load_triggered(){

	TFAbstractHolder* loaded = TFHolderFactory::loadHolder(this);
	if(!loaded) return;
	
	addToPalette_(loaded);

	actionSave_->setEnabled(true);
}

void TFWindow::newTF_triggered(const TFHolderType &tfType){

	TFAbstractHolder* holder = TFHolderFactory::createHolder(this, tfType);

	if(!holder){
		QMessageBox::warning(this, QObject::tr("Transfer Functions"), QObject::tr("Creating error."));
		return;
	}

	addToPalette_(holder);

	actionSave_->setEnabled(true);
}

void TFWindow::change_holder(const TFSize &index, const bool& forceChange){

	if(!forceChange && index == activeHolder_) return;

	if(index < 0 || index >= palette_.size())
	{
		tfAssert(!"palette out of range");
		return;
	}

	if(activeHolder_ >= 0) palette_[activeHolder_]->hideHolder();

	activeHolder_ = index;

	palette_[activeHolder_]->showHolder();
}

} // namespace GUI
} // namespace M4D
