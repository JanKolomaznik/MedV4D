#ifndef TF_RGB_HOLDER
#define TF_RGB_HOLDER

#include <TFAbstractHolder.h>
#include <TFRGBaFunction.h>
#include <TFRGBPainter.h>

namespace M4D {
namespace GUI {

class TFRGBHolder: public TFAbstractHolder{

public:

	TFRGBHolder(QMainWindow* parent);
	~TFRGBHolder();

	//void setUp(TFSize index);

protected:

	void updateFunction_();
	void updatePainter_();
	void resizePainter_();

	TFAbstractFunction* getFunction_();

private:

	TFRGBaFunction function_;
	TFRGBPainter painter_;
};

} // namespace GUI
} // namespace M4D

#endif //TF_RGB_HOLDER