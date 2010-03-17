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

		_save(file);

		file.close();
	}

	virtual void setup(QWidget *parent, const QRect rect) = 0;

	TFType getType() const{
		return _type;
	}

protected slots:
    virtual void on_use_clicked() = 0;

signals:
	void UseTransferFunction(TFAbstractFunction &transferFunction);

protected:
	TFType _type;

	Ui::TFAbstractHolder* _basicTools;

	virtual bool _load(QFile &file) = 0;

	virtual void _save(QFile &file) = 0;

	TFAbstractHolder(): _basicTools(new Ui::TFAbstractHolder), _type(TFTYPE_UNKNOWN){

		_basicTools->setupUi(this);
	}
};
#endif //TF_ABSTRACTHOLDER