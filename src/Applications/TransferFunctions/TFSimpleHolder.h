#ifndef TF_SIMPLEHOLDER
#define TF_SIMPLEHOLDER

#include "common/Types.h"

#include <TFAbstractHolder.h>
#include <TFSimpleFunction.h>
#include <TFSimplePainter.h>
#include <TFXmlSimpleReader.h>
#include <TFXmlSimpleWriter.h>

#include <string>
#include <map>
#include <vector>

#include <QtGui/QWidget>


class TFSimpleHolder: public TFAbstractHolder{

    Q_OBJECT

public:
	TFSimpleHolder();
	virtual ~TFSimpleHolder();

	virtual void setup(QWidget *parent, const QRect rect);

protected slots:
    virtual void on_use_clicked();

protected:
	virtual void _save(QFile &file);
	virtual bool _load(QFile &file);

private:
	TFSimpleFunction _function;
	TFSimplePainter _painter;
};
#endif //TF_SIMPLEHOLDER