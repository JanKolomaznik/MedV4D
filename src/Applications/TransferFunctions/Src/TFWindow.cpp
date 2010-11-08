#include "TFWindow.h"

namespace M4D {
namespace GUI {

TFWindow::TFWindow(): ui_(new Ui::TFWindow), holder_(NULL){

    ui_->setupUi(this);
}

TFWindow::~TFWindow(){
}

void TFWindow::setupDefault(){
	
	holder_ = TFHolderFactory::createHolder(this, TFTYPE_SIMPLE);

	if(!holder_){
		QMessageBox::warning(this, QObject::tr("Transfer Functions"), QObject::tr("Creating error."));
		return;
	}

	actionSave_->setEnabled(true);
	setupHolder();
}

void TFWindow::setupHolder(){

	holder_->setUp(this, QRect(0, 0, width(), height()));
	
	QObject::connect( holder_, SIGNAL(UseTransferFunction(TFAbstractFunction&)), this, SLOT(modify_data(TFAbstractFunction&)));
	
	QObject::connect( this, SIGNAL(ResizeHolder(const QRect)), holder_, SLOT(size_changed(const QRect)));
}

void TFWindow::createMenu(QMenuBar* menubar){

	//menu->setTitle(QString::fromStdString("Transfer Function"));
	menuTF_ = new QMenu(QString::fromStdString("Transfer Function"), menubar);

	//menu new
	menuNew_ = new QMenu(QString::fromStdString("New"), menuTF_);	
	tfActions_ = TFHolderFactory::createMenuTFActions(menuNew_);

	TFActionsIt begin = tfActions_.begin();
	TFActionsIt end = tfActions_.end();
	for(TFActionsIt it = begin; it!=end; ++it)
	{
		bool menuActionConnected = QObject::connect( *it, SIGNAL(TFActionClicked(TFType&)), this, SLOT(newTF_triggered(TFType&)));
		tfAssert(menuActionConnected);
		menuNew_->addAction(*it);
	}

	menuTF_->addAction(menuNew_->menuAction());

	//action load
	actionLoad_ = new QAction(QString::fromStdString("Load"), menuTF_);
	bool loadConnected = QObject::connect( actionLoad_, SIGNAL(triggered()), this, SLOT(on_load_triggered()));
	tfAssert(loadConnected);
	menuTF_->addAction(actionLoad_);

	//action save
	actionSave_ = new QAction(QString::fromStdString("Save"), menuTF_);
	actionSave_->setEnabled(false);
	bool saveConnected = QObject::connect( actionSave_, SIGNAL(triggered()), this, SLOT(on_save_triggered()));
	tfAssert(saveConnected);
	menuTF_->addAction(actionSave_);

	//separator
	menuTF_->addSeparator();	

	//action exit
	actionExit_ = new QAction(QString::fromStdString("Exit"), menuTF_);
	bool exitConnected = QObject::connect( actionExit_, SIGNAL(triggered()), this, SLOT(on_exit_triggered()));
	tfAssert(exitConnected);
	menuTF_->addAction(actionExit_);
	
	menuTF_->setEnabled(true);
	menubar->addAction(menuTF_->menuAction());
}

void TFWindow::resizeEvent(QResizeEvent* e){

	emit ResizeHolder(QRect(0, 0, width(), height()));
}

void TFWindow::on_exit_triggered(){

    close();
}

void TFWindow::on_save_triggered(){

	holder_->save();
}

void TFWindow::on_load_triggered(){

	TFAbstractHolder* loaded = NULL;
	loaded = TFHolderFactory::loadHolder(this);
	if(!loaded) return;
	
	if(holder_)
	{
		holder_->hide();
		delete holder_;
	}
	holder_ = loaded;
	actionSave_->setEnabled(true);
	setupHolder();
}

void TFWindow::newTF_triggered(TFType &tfType){

	if(holder_)
	{
		delete holder_;
		holder_ = NULL;
	}

	if(holder_) holder_->hide();
	holder_ = TFHolderFactory::createHolder(this, tfType);

	if(!holder_){
		QMessageBox::warning(this, QObject::tr("Transfer Functions"), QObject::tr("Creating error."));
		return;
	}

	actionSave_->setEnabled(true);
	setupHolder();
}

void TFWindow::on_menuHistogram_triggered(){

}

/*
void TFWindow::receive_histogram(const TFHistogram& histogram){
	
	holder_->receiveHistogram(histogram);
}
*/

} // namespace GUI
} // namespace M4D
