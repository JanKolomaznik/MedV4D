#ifndef TF_HSVA_HOLDER
#define TF_HSVA_HOLDER

#include <TFAbstractHolder.h>
#include <TFHSVaFunction.h>
#include <TFHSVaPainter.h>

namespace M4D {
namespace GUI {

class TFHSVaHolder: public TFAbstractHolder{

public:

	TFHSVaHolder(QMainWindow* parent);
	~TFHSVaHolder();

	//void setUp(TFSize index);

protected:

	void updateFunction_();
	void updatePainter_();
	void resizePainter_();

	TFAbstractFunction* getFunction_();

private:

	TFHSVaFunction function_;
	TFHSVaPainter painter_;
};

} // namespace GUI
} // namespace M4D

#endif //TF_HSVA_HOLDER