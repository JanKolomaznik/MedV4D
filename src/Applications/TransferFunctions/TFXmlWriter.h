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
	void write(QIODevice* device, TFAbstractFunction* data, const TFHolderType& holderType);
	void writeTestData(QIODevice* device, const TFHolderType& holderType);

private:
	void writeFunction_(TFAbstractFunction* function, const TFHolderType& holderType);
	void writePoint_(TFColor point);
};

} // namespace GUI
} // namespace M4D

#endif	//TF_XMLWRITER