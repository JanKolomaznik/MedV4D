#ifndef TF_HSV_HOLDER
#define TF_HSV_HOLDER

#include <TFAbstractHolder.h>
#include <TFHSVaFunction.h>
#include <TFHSVPainter.h>

namespace M4D {
namespace GUI {

class TFHSVHolder: public TFAbstractHolder{

public:

	TFHSVHolder(QWidget* window);
	~TFHSVHolder();

	void setUp(const TFSize& index);

protected:

	void updateFunction_();
	void updatePainter_();
	void resizePainter_(const QRect& rect);

	TFAbstractFunction* getFunction_();

private:

	TFHSVaFunction function_;
	TFHSVPainter painter_;
};

} // namespace GUI
} // namespace M4D

#endif //TF_HSV_HOLDER