#ifndef TF_GRAYSCALEALPHA_HOLDER
#define TF_GRAYSCALEALPHA_HOLDER

#include <TFAbstractHolder.h>
#include <TFRGBaFunction.h>
#include <TFGrayscaleAlphaPainter.h>

namespace M4D {
namespace GUI {

class TFGrayscaleAlphaHolder: public TFAbstractHolder{

public:

	TFGrayscaleAlphaHolder(QMainWindow* parent);
	~TFGrayscaleAlphaHolder();

	//void setUp(TFSize index);

protected:

	void updateFunction_();
	void updatePainter_();
	void resizePainter_();

	TFAbstractFunction* getFunction_();

private:

	TFRGBaFunction function_;
	TFGrayscaleAlphaPainter painter_;
};

} // namespace GUI
} // namespace M4D

#endif //TF_GRAYSCALEALPHA_HOLDER