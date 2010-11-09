#ifndef TF_GRAYSCALE_XMLWRITER
#define TF_GRAYSCALE_XMLWRITER

#include <QtCore/QXmlStreamWriter>
#include <QtCore/QString>

#include <TFGrayscaleFunction.h>

namespace M4D {
namespace GUI {

class TFGrayscaleXmlWriter : public QXmlStreamWriter
{
public:
	TFGrayscaleXmlWriter();
	void write(QIODevice* device, TFGrayscaleFunction &data);
	void writeTestData(QIODevice* device);

private:
	void writeFunction(TFGrayscaleFunction &function);
	void writePoint(float point);
};

} // namespace GUI
} // namespace M4D

#endif	//TF_GRAYSCALE_XMLWRITER