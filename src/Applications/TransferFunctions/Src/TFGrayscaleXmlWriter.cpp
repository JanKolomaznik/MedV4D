#include "TFGrayscaleXmlWriter.h"

namespace M4D {
namespace GUI {

TFGrayscaleXmlWriter::TFGrayscaleXmlWriter(){
	setAutoFormatting(true);
}

void TFGrayscaleXmlWriter::write(QIODevice *device, TFGrayscaleFunction &data){

	setDevice(device);

	writeStartDocument();
	writeDTD("<!DOCTYPE TransferFunctions>");

	writeFunction(data);

	writeEndDocument();
}

void TFGrayscaleXmlWriter::writeTestData(QIODevice *device){
	TFGrayscaleFunction data;
	TFFunctionMapPtr f = data.getFunction();
	TFSize domain = data.getDomain();
	for(TFSize i = 0; i < domain; ++i)
	{
		(*f)[i] = (i/4)/1000.0;
	}
	write(device, data);
}

void TFGrayscaleXmlWriter::writeFunction(TFGrayscaleFunction &function){

	writeStartElement("TransferFunction");
	writeAttribute("type", QString::fromStdString(convert<TFType, std::string>(function.getType())));
	writeAttribute("domain", QString::fromStdString( convert<TFSize, std::string>(function.getDomain()) ));

	const TFFunctionMapPtr points = function.getFunction();
	TFFunctionMap::const_iterator first = points->begin();
	TFFunctionMap::const_iterator end = points->end();
	for(TFFunctionMap::const_iterator it = first; it != end; ++it)
	{
		writePoint(*it);
	}

	writeEndElement();
}

void TFGrayscaleXmlWriter::writePoint(float point){

	writeStartElement("TFPaintingPoint");
	writeAttribute("point", QString::fromStdString( convert<float, std::string>(point)) );

	writeEndElement();
}

} // namespace GUI
} // namespace M4D
