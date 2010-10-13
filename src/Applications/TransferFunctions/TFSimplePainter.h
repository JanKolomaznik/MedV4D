#ifndef TF_SIMPLEPAINTER
#define TF_SIMPLEPAINTER

#include <QtGui/QWidget>
#include <QtGui/QPainter>
#include <QtGui/QMouseEvent>
#include <QtGui/QPaintEvent>

#include <vector>

#include <TFSimpleFunction.h>

namespace Ui{

	class TFSimplePainter;
}

class TFSimplePainter: public QWidget{

	Q_OBJECT

public:
	TFSimplePainter();

	TFSimplePainter(int marginH, int marginV);

	~TFSimplePainter();

	void setUp(QWidget *parent);

	void setUp(QWidget *parent, int marginH, int marginV);

	void resize(const QRect rect);

	void setView(TFPointMap view);

	void setAutoUpdate(bool state);

	TFPointMap getView();

	TFPoints getPoints();

	void setHistogram(const TFHistogram& histogram);
	void paintHistogram(const bool paint);

signals:
	void FunctionChanged();

protected:
	void paintEvent(QPaintEvent *e);
	void mousePressEvent(QMouseEvent *e);
	void mouseReleaseEvent(QMouseEvent *e);
	void mouseMoveEvent(QMouseEvent *e);

private:
	Ui::TFSimplePainter* painter_;
	int marginV_, marginH_;
	TFPointMap view_;	
	TFPoint* drawHelper_;
	bool autoUpdate_;

	TFPoint painterCoords(const TFPoint &point);

	TFPoint painterCoords(int x, int y);

	void addLine(int x1, int y1, int x2, int y2);
	void addPoint(TFPoint point);
};

#endif //TF_SIMPLEPAINTER