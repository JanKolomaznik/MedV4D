#ifndef TF_RGB_HOLDER
#define TF_RGB_HOLDER

#include "common/Types.h"

#include <TFAbstractHolder.h>
#include <TFRGBFunction.h>
#include <TFRGBPainter.h>
//#include <TFGrayscaleXmlREADER.h>
//#include <TFGrayscaleXmlWriter.h>

#include <string>
#include <map>
#include <vector>

namespace M4D {
namespace GUI {

class TFRGBHolder: public TFAbstractHolder{

public:
	TFRGBHolder(QWidget* window);
	~TFRGBHolder();

	void setUp(QWidget *parent, const QRect rect);

protected:
	void save_(QFile &file);
	bool load_(QFile &file);

	void updateFunction_();
	void updatePainter_(const QRect& rect);

	TFAbstractFunction* getFunction_();

private:
	TFRGBFunction function_;
	TFRGBPainter painter_;
};

} // namespace GUI
} // namespace M4D

#endif //TF_RGB_HOLDER