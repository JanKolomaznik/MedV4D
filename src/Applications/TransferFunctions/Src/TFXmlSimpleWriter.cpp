#include "TFXmlSimpleWriter.h"

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

void TFXmlSimpleWriter::writeFunction(TFSimpleFunction &function){

	writeStartElement("TransferFunction");
	writeAttribute("type", QString::fromStdString(convert<TFType, std::string>(function.getType())));
	writeAttribute("name", QString::fromStdString(function.name));

	TFPoints points = function.getAllPoints();
	TFPointsIterator first = points.begin();
	TFPointsIterator end = points.end();
	TFPointsIterator it = first;

	for(it; it != end; ++it)
	{
		writePoint(*it);
	}

	writeEndElement();
}

void TFXmlSimpleWriter::writePoint(TFPoint &point){

	writeStartElement("TFPoint");
	writeAttribute("x", QString::fromStdString( convert<int, std::string>(point.x)) );
	writeAttribute("y", QString::fromStdString( convert<int, std::string>(point.y)) );

	writeEndElement();
}