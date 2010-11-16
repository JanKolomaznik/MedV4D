#ifndef TF_RGBA_HOLDER
#define TF_RGBA_HOLDER

#include "common/Types.h"

#include <TFAbstractHolder.h>
#include <TFRGBaFunction.h>
#include <TFRGBaPainter.h>

namespace M4D {
namespace GUI {

class TFRGBaHolder: public TFAbstractHolder{

public:
	TFRGBaHolder(QWidget* window);
	~TFRGBaHolder();

	void setUp(QWidget *parent, const QRect& rect);

protected:
	void updateFunction_();
	void updatePainter_();
	void resizePainter_(const QRect& rect);

	TFAbstractFunction* getFunction_();

private:
	TFRGBaFunction function_;
	TFRGBaPainter painter_;
};

} // namespace GUI
} // namespace M4D

#endif //TF_RGBA_HOLDER