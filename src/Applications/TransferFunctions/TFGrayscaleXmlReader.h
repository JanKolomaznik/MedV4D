#ifndef TF_GRAYSCALE_XMLREADER
#define TF_GRAYSCALE_XMLREADER

#include <QtCore/QXmlStreamReader>
#include <QtCore/QString>
#include <QtGui/QMessageBox>

#include <TFGrayscaleFunction.h>

namespace M4D {
namespace GUI {

class TFGrayscaleXmlREADER: public QXmlStreamReader{

public:
	TFGrayscaleXmlREADER();
	~TFGrayscaleXmlREADER();

	void read(QIODevice* device, TFGrayscaleFunction* function, bool &error);
	void readTestData(TFGrayscaleFunction* function);

private:
	void readFunction(TFGrayscaleFunction* function, bool &error);
	float readPoint(bool &error);
};

} // namespace GUI
} // namespace M4D

#endif //TF_GRAYSCALE_XMLREADER