#ifndef TF_GRAYSCALETRANSPARENCY_HOLDER
#define TF_GRAYSCALETRANSPARENCY_HOLDER

#include "common/Types.h"

#include <TFAbstractHolder.h>
#include <TFGrayscaleTransparencyFunction.h>
#include <TFGrayscaleTransparencyPainter.h>
//#include <TFXmlSimpleReader.h>
//#include <TFXmlSimpleWriter.h>

#include <string>
#include <map>
#include <vector>

namespace M4D {
namespace GUI {

#define PAINTER_X 25
#define PAINTER_Y 25
#define PAINTER_MARGIN 5

class TFGrayscaleTransparencyHolder: public TFAbstractHolder{

public:
	TFGrayscaleTransparencyHolder(QWidget* window);
	~TFGrayscaleTransparencyHolder();

	void setUp(QWidget *parent, const QRect rect);

protected slots:
	void size_changed(const QRect rect);

protected:
	void save_(QFile &file);
	bool load_(QFile &file);
	void updateFunction_();

	TFAbstractFunction* getFunction_();

private:
	TFGrayscaleTransparencyFunction function_;
	TFGrayscaleTransparencyPainter painter_;
};

} // namespace GUI
} // namespace M4D

#endif //TF_GRAYSCALETRANSPARENCY_HOLDER