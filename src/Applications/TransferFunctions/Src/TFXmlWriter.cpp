#include <TFXmlWriter.h>

namespace M4D {
namespace GUI {

TFXmlWriter::TFXmlWriter(QFile *file):
	qWriter_(file),
	elementDepth_(0){

	qWriter_.setAutoFormatting(true);
	qWriter_.writeStartDocument();
}

TFXmlWriter::~TFXmlWriter(){}

void TFXmlWriter::writeDTD(const std::string& attribute){

	qWriter_.writeDTD(QString::fromStdString("<!DOCTYPE " + attribute + ">"));
}

void TFXmlWriter::writeStartElement(const std::string& element){

	qWriter_.writeStartElement(QString::fromStdString(element));
	++elementDepth_;
}

void TFXmlWriter::writeEndElement(){

	qWriter_.writeEndElement();
	--elementDepth_;
}

void TFXmlWriter::writeAttribute(const std::string& attribute, const std::string& value){

	qWriter_.writeAttribute(QString::fromStdString(attribute), QString::fromStdString(value));
}

void TFXmlWriter::finalizeDocument(){

	while(elementDepth_ > 0) writeEndElement();
	qWriter_.writeEndDocument();
}

} // namespace GUI
} // namespace M4D
