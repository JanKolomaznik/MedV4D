#ifndef TF_XMLWRITER
#define TF_XMLWRITER

#include <QtCore/QXmlStreamWriter>
#include <QtCore/QString>

#include <TFAbstractFunction.h>

namespace M4D {
namespace GUI {

class TFXmlWriter : public QXmlStreamWriter
{
public:
	TFXmlWriter();
	void write(QIODevice* device, TFAbstractFunction::Ptr data);
	void writeTestData(QIODevice* device);

private:
	void writeFunction_(TFAbstractFunction::Ptr function);
	void writePoint_(TFColor point);
};

} // namespace GUI
} // namespace M4D

#endif	//TF_XMLWRITER