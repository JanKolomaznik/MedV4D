#include "TFXmlSimpleWriter.h"

namespace M4D {
namespace GUI {

TFXmlSimpleWriter::TFXmlSimpleWriter(){
	setAutoFormatting(true);
}

void TFXmlSimpleWriter::write(QIODevice *device, TFSimpleFunction &data){

	setDevice(device);

	writeStartDocument();
	writeDTD("<!DOCTYPE TransferFunctions>");

	writeFunction(data);

	writeEndDocument();
}

void TFXmlSimpleWriter::writeTestData(QIODevice *device){
	TFSimpleFunction data;
	TFFunctionMapPtr f = data.getFunction();
	TFSize domain = data.getDomain();
	for(TFSize i = 0; i < domain; ++i)
	{
		(*f)[i] = (i/4)/1000.0;
	}
	write(device, data);
}

void TFXmlSimpleWriter::writeFunction(TFSimpleFunction &function){

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

void TFXmlSimpleWriter::writePoint(float point){

	writeStartElement("TFPointSimple");
	writeAttribute("point", QString::fromStdString( convert<float, std::string>(point)) );

	writeEndElement();
}

} // namespace GUI
} // namespace M4D
