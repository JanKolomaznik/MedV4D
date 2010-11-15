#ifndef TF_RGB_HOLDER
#define TF_RGB_HOLDER

#include "common/Types.h"

#include <TFAbstractHolder.h>
#include <TFRGBaFunction.h>
#include <TFRGBPainter.h>

namespace M4D {
namespace GUI {

class TFRGBHolder: public TFAbstractHolder{

public:
	TFRGBHolder(QWidget* window);
	~TFRGBHolder();

	void setUp(QWidget *parent, const QRect rect);

protected:
	void updateFunction_();
	void updatePainter_();
	void resizePainter_(const QRect& rect);

	TFAbstractFunction* getFunction_();

private:
	TFRGBaFunction function_;
	TFRGBPainter painter_;
};

} // namespace GUI
} // namespace M4D

#endif //TF_RGB_HOLDER