#ifndef TF_XMLSIMPLEWRITER
#define TF_XMLSIMPLEWRITER

#include <QtCore/QXmlStreamWriter>
#include <QtCore/QString>

#include <TFSimpleFunction.h>

namespace M4D {
namespace GUI {

class TFXmlSimpleWriter : public QXmlStreamWriter
{
public:
	TFXmlSimpleWriter();
	void write(QIODevice* device, TFSimpleFunction &data);
	void writeTestData(QIODevice* device);

private:
	void writeFunction(TFSimpleFunction &function);
	void writePoint(float point);
};

} // namespace GUI
} // namespace M4D

#endif	//TF_XMLSIMPLEWRITER