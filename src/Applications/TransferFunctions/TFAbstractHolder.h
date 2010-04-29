#ifndef TF_ABSTRACTHOLDER
#define TF_ABSTRACTHOLDER

#include "common/Types.h"
#include "ui_TFAbstractHolder.h"

#include <QtGui/QWidget>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>

#include <QtCore/QString>

#include <TFTypes.h>
#include <TFAbstractFunction.h>

namespace Ui{

    class TFAbstractHolder;
}

class TFAbstractHolder : public QWidget{

	Q_OBJECT
	
	friend class TFHolderFactory;

public:
	virtual ~TFAbstractHolder(){}

	virtual void save(){

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

	virtual void setUp(QWidget *parent, const QRect rect) = 0;

	TFType getType() const{
		return type_;
	}

protected slots:
    virtual void on_use_clicked() = 0;
	virtual void size_changed(const QRect rect) = 0;
	virtual void on_autoUpdate_stateChanged(int state) = 0;

signals:
	void UseTransferFunction(TFAbstractFunction &transferFunction);

protected:
	TFType type_;
	Ui::TFAbstractHolder* basicTools_;
	bool setup_;

	virtual bool load_(QFile &file) = 0;
	virtual void save_(QFile &file) = 0;

	TFAbstractHolder(): basicTools_(new Ui::TFAbstractHolder), type_(TFTYPE_UNKNOWN), setup_(false){

		basicTools_->setupUi(this);
	}
};
#endif //TF_ABSTRACTHOLDER