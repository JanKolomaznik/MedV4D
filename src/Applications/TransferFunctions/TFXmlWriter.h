#ifndef TF_XMLWRITER
#define TF_XMLWRITER

#include <QtCore/QXmlStreamWriter>
#include <QtCore/QString>
#include <QtCore/QFile>

#include <TFCommon.h>

namespace M4D {
namespace GUI {

class TFXmlWriter{

public:

	typedef boost::shared_ptr<TFXmlWriter> Ptr;

	TFXmlWriter(QFile* file);
	~TFXmlWriter();

	void writeDTD(const std::string dtd);

	void writeStartElement(const std::string element);
	void writeEndElement();

	void writeAttribute(const std::string attribute, const std::string value);

	void finalizeDocument();

private:

	QXmlStreamWriter qWriter_;
	TF::Size elementDepth_;
};

} // namespace GUI
} // namespace M4D

#endif	//TF_XMLWRITER