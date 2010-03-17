#ifndef TF_XMLSIMPLEWRITER
#define TF_XMLSIMPLEWRITER

#include <QtCore/QXmlStreamWriter>
#include <QtCore/QString>

#include <TFSimpleFunction.h>

class TFXmlSimpleWriter : public QXmlStreamWriter
{
public:
	TFXmlSimpleWriter();
	void write(QIODevice* device, TFSimpleFunction &data);

private:
	void writeFunction(TFSimpleFunction &function);
	void writePoint(TFPoint &point);
};

#endif	//TF_XMLSIMPLEWRITER