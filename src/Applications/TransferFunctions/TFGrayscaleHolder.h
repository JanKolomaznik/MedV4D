#ifndef TF_GRAYSCALE_HOLDER
#define TF_GRAYSCALE_HOLDER

#include <TFAbstractHolder.h>
#include <TFRGBaFunction.h>
#include <TFGrayscalePainter.h>

namespace M4D {
namespace GUI {

class TFGrayscaleHolder: public TFAbstractHolder{

public:
	TFGrayscaleHolder(QWidget* window);
	~TFGrayscaleHolder();

	void setUp(QWidget *parent, const QRect rect);

protected:
	void updateFunction_();
	void updatePainter_();
	void resizePainter_(const QRect& rect);

	TFAbstractFunction* getFunction_();

private:
	TFRGBaFunction function_;
	TFGrayscalePainter painter_;
};

} // namespace GUI
} // namespace M4D

#endif //TF_GRAYSCALE_HOLDER