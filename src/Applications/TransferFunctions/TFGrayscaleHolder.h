#ifndef TF_GRAYSCALE_HOLDER
#define TF_GRAYSCALE_HOLDER

#include "common/Types.h"

#include <TFAbstractHolder.h>
#include <TFGrayscaleFunction.h>
#include <TFGrayscalePainter.h>
#include <TFGrayscaleXmlREADER.h>
#include <TFGrayscaleXmlWriter.h>

#include <string>
#include <map>
#include <vector>

namespace M4D {
namespace GUI {

class TFGrayscaleHolder: public TFAbstractHolder{

public:
	TFGrayscaleHolder(QWidget* window);
	~TFGrayscaleHolder();

	void setUp(QWidget *parent, const QRect rect);

protected:
	void save_(QFile &file);
	bool load_(QFile &file);

	void updateFunction_();
	void updatePainter_(const QRect& rect);

	TFAbstractFunction* getFunction_();

private:
	TFGrayscaleFunction function_;
	TFGrayscalePainter painter_;
};

} // namespace GUI
} // namespace M4D

#endif //TF_GRAYSCALE_HOLDER