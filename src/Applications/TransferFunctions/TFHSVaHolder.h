#ifndef TF_HSVA_HOLDER
#define TF_HSVA_HOLDER

#include "common/Types.h"

#include <TFAbstractHolder.h>
#include <TFHSVaFunction.h>
#include <TFHSVaPainter.h>

namespace M4D {
namespace GUI {

class TFHSVaHolder: public TFAbstractHolder{

public:
	TFHSVaHolder(QWidget* window);
	~TFHSVaHolder();

	void setUp(QWidget *parent, const QRect rect);

protected:
	void updateFunction_();
	void updatePainter_();
	void resizePainter_(const QRect& rect);

	TFAbstractFunction* getFunction_();

	void paintEvent(QPaintEvent *e);

private:

	TFHSVaFunction function_;
	TFHSVaPainter painter_;

	TFSize colorBarWidth_;
	TFSize painterMargin_;
};

} // namespace GUI
} // namespace M4D

#endif //TF_HSVA_HOLDER