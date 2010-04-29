
#include "TFSimpleHolder.h"

TFSimpleHolder::TFSimpleHolder(): autoUpdate_(false){

	type_ = TFTYPE_SIMPLE;

	basicTools_->name->setText(QString::fromStdString(function_.name));
}

TFSimpleHolder::~TFSimpleHolder(){

	if(basicTools_) delete basicTools_;
}

void TFSimpleHolder::setUp(QWidget *parent, const QRect rect){

	painter_.setUp(this);
	size_changed(rect);
	setParent(parent);
	show();

	QObject::connect( &painter_, SIGNAL(FunctionChanged()), this, SLOT(on_use_clicked()));
}

void TFSimpleHolder::save_(QFile &file){

	function_.setPoints(painter_.getView());

	function_.name = basicTools_->name->text().toStdString();

	 TFXmlSimpleWriter writer;
     writer.write(&file, function_);
}

bool TFSimpleHolder::load_(QFile &file){

	TFXmlSimpleReader reader;

	bool error = false;

	TFSimpleFunction loaded = reader.read(&file, error);

	if (error || reader.error())
	{
		return false;
	}

	function_ = loaded;

	painter_.setView(function_.getPointMap());

	basicTools_->name->setText(QString::fromStdString(function_.name));

	return true;
}

void TFSimpleHolder::on_use_clicked(){

	function_.setPoints(painter_.getView());

	emit UseTransferFunction(function_);
}

void TFSimpleHolder::on_autoUpdate_stateChanged(int state){

	autoUpdate_ = (state == Qt::Checked);

	basicTools_->use->setEnabled(!autoUpdate_);
	painter_.setAutoUpdate(autoUpdate_);

	if(autoUpdate_) on_use_clicked();
}

void TFSimpleHolder::size_changed(const QRect rect){

	setGeometry(rect);

	int newWidth = rect.width() - 2*PAINTER_X;
	int newHeight = rect.height() - 2*PAINTER_Y;

	TFPointMap painterFunction = painter_.getView();

	if(!painterFunction.empty())
	{
		function_.setPoints(painterFunction);
	}
	function_.recalculate(newWidth - 2*PAINTER_MARGIN_H, newHeight - 2*PAINTER_MARGIN_V);

	painter_.resize(QRect(PAINTER_X, PAINTER_Y, newWidth, newHeight));
	painter_.setView(function_.getPointMap());

	if(autoUpdate_) on_use_clicked();
}