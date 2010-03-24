#include "TFWindow.h"
#include "ui_TFWindow.h"

#include <cassert>

#include <TFSimpleHolder.h>

/*
 * constructor, destructor
 */

TFWindow::TFWindow(): ui(new Ui::TFWindow), _holder(NULL){

    ui->setupUi(this);
}

TFWindow::~TFWindow(){

	for(unsigned i = 0; i < tfActions.size(); ++i)
	{
		delete tfActions[i];
	}

	if(_holder) delete _holder;
    if(ui) delete ui;
}

void TFWindow::build(){

	tfActions = TFHolderFactory::createMenuTFActions(this, ui->menuNew);

	for(int i = 0; i < tfActions.size(); ++i)
	{
		 bool ok = QObject::connect( tfActions[i], SIGNAL(TFActionClicked(TFType&)), this, SLOT(on_newTF_triggered(TFType&)));
	}
}

void TFWindow::setupHolder(){

	_holder->setup(this, QRect(0, MENU_SPACE, width(), height() - MENU_SPACE));
	QObject::connect( _holder, SIGNAL(UseTransferFunction(TFAbstractFunction&)), this, SLOT(modify_data(TFAbstractFunction&)));
	QObject::connect( this, SIGNAL(ResizeHolder(const QRect)), _holder, SLOT(size_changed(const QRect)));
}

void TFWindow::resizeEvent(QResizeEvent *event){

	ui->menuBar->setGeometry(QRect(0, 0, width(), ui->menuBar->height()));
	emit ResizeHolder(QRect(0, MENU_SPACE, width(), height() - MENU_SPACE));
}

void TFWindow::on_exit_triggered(){

    close();
}

void TFWindow::on_save_triggered(){

	_holder->save();
}

void TFWindow::on_load_triggered(){

	_holder = TFHolderFactory::load(this);
	if(_holder) setupHolder();
	//TODO else error messagebox
}

void TFWindow::on_newTF_triggered(TFType &tfType){

	if(_holder)
	{
		delete _holder;
		_holder = NULL;
	}
	_holder = TFHolderFactory::create(tfType);
	if(_holder) setupHolder();
	//TODO else error messagebox
}

void TFWindow::modify_data(TFAbstractFunction &transferFunction){

	emit AdjustByTransferFunction(transferFunction);
}
