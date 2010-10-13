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
	TFSimpleHolder(TFWindowI* window);
	~TFSimpleHolder();

	void setUp(QWidget *parent, const QRect rect);
	void receiveHistogram(const TFHistogram& histogram);

protected slots:
    void on_use_clicked();
	void size_changed(const QRect rect);
	void on_autoUpdate_stateChanged(int state);

protected:
	void save_(QFile &file);
	bool load_(QFile &file);

private:
	TFSimpleFunction function_;
	TFSimplePainter painter_;
	bool autoUpdate_;
};
#endif //TF_SIMPLEHOLDER