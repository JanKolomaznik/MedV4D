#ifndef TF_SCHEMEPAINTER
#define TF_SCHEMEPAINTER

#include <QtGui/QWidget>
#include <QtGui/QPainter>
#include <QtGui/QMouseEvent>
#include <QtGui/QPaintEvent>
#include <TF/TFScheme.h>

#include <vector>

#include <TF/TFSchemeFunction.h>

namespace Ui{

	class TFSchemePainter;
}

class TFSchemePainter: public QWidget{

	Q_OBJECT

public:
	TFSchemePainter(int marginH, int marginV);

	~TFSchemePainter();

	void setView(TFScheme* scheme);

protected:
	void paintEvent(QPaintEvent *e);
	void mousePressEvent(QMouseEvent *e);
	void mouseMoveEvent(QMouseEvent *e);

private slots:
	void Repaint();

private:
	Ui::TFSchemePainter* painter;

	int _marginV, _marginH;
	TFScheme* currentView;
};

#endif //TF_SCHEMEPAINTER