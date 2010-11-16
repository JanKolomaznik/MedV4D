#ifndef TF_HSV_HOLDER
#define TF_HSV_HOLDER

#include "common/Types.h"

#include <TFAbstractHolder.h>
#include <TFHSVaFunction.h>
#include <TFHSVPainter.h>

namespace M4D {
namespace GUI {

class TFHSVHolder: public TFAbstractHolder{

public:
	TFHSVHolder(QWidget* window);
	~TFHSVHolder();

	void setUp(QWidget *parent, const QRect& rect);

protected:
	void updateFunction_();
	void updatePainter_();
	void resizePainter_(const QRect& rect);

	TFAbstractFunction* getFunction_();

	void paintEvent(QPaintEvent *e);

private:

	TFHSVaFunction function_;
	TFHSVPainter painter_;

	TFSize colorBarWidth_;
	TFSize painterMargin_;
};

} // namespace GUI
} // namespace M4D

#endif //TF_HSV_HOLDER