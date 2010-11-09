#ifndef TF_GRAYSCALETRANSPARENCY_HOLDER
#define TF_GRAYSCALETRANSPARENCY_HOLDER

#include "common/Types.h"

#include <TFAbstractHolder.h>
#include <TFGrayscaleTransparencyFunction.h>
#include <TFGrayscaleTransparencyPainter.h>
//#include <TFGrayscaleXmlREADER.h>
//#include <TFGrayscaleXmlWriter.h>

#include <string>
#include <map>
#include <vector>

namespace M4D {
namespace GUI {

class TFGrayscaleTransparencyHolder: public TFAbstractHolder{

public:
	TFGrayscaleTransparencyHolder(QWidget* window);
	~TFGrayscaleTransparencyHolder();

	void setUp(QWidget *parent, const QRect rect);

protected:
	void save_(QFile &file);
	bool load_(QFile &file);

	void updateFunction_();
	void updatePainter_(const QRect& rect);

	TFAbstractFunction* getFunction_();

private:
	TFGrayscaleTransparencyFunction function_;
	TFGrayscaleTransparencyPainter painter_;
};

} // namespace GUI
} // namespace M4D

#endif //TF_GRAYSCALETRANSPARENCY_HOLDER