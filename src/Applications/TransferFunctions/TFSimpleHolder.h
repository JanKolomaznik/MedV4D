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


#define PAINTER_X 30
#define PAINTER_Y 70
#define PAINTER_MARGIN_H 10
#define PAINTER_MARGIN_V 10


class TFSimpleHolder: public TFAbstractHolder{

    Q_OBJECT

public:
	TFSimpleHolder();
	~TFSimpleHolder();

	void setup(QWidget *parent, const QRect rect);

protected slots:
    void on_use_clicked();
	void size_changed(const QRect rect);

protected:
	void _save(QFile &file);
	bool _load(QFile &file);

private:
	TFSimpleFunction _function;
	TFSimplePainter _painter;
};
#endif //TF_SIMPLEHOLDER