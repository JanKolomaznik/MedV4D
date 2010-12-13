#include "TFXmlWriter.h"

#include <TFRGBaFunction.h>

namespace M4D {
namespace GUI {

TFXmlWriter::TFXmlWriter(){
	setAutoFormatting(true);
}

void TFXmlWriter::write(QIODevice *device, TFAbstractFunction* data, TFHolderType holderType){

	setDevice(device);

	writeStartDocument();
		writeDTD("<!DOCTYPE TransferFunctions>");

		writeFunction_(data, holderType);

	writeEndDocument();
}

void TFXmlWriter::writeTestData(QIODevice *device, TFHolderType holderType){
	
	TFRGBaFunction data;
	TFColorMapPtr f = data.getColorMap();
	TFSize domain = data.getDomain();
	for(TFSize i = 0; i < domain; ++i)
	{
		(*f)[i].component1 = (i/4)/1000.0;
		(*f)[i].component2 = (i/4)/1000.0;
		(*f)[i].component3 = (i/4)/1000.0;
		(*f)[i].alpha = (i/4)/1000.0;
	}
	write(device, &data, holderType);	
}

void TFXmlWriter::writeFunction_(TFAbstractFunction* function, TFHolderType holderType){
	
	writeStartElement("TransferFunction");
		writeAttribute("holderType", QString::fromStdString(convert<TFHolderType, std::string>(holderType)));
		writeAttribute("functionType", QString::fromStdString(convert<TFFunctionType, std::string>(function->getType())));
		writeAttribute("domain", QString::fromStdString( convert<TFSize, std::string>(function->getDomain()) ));

		const TFColorMapPtr points = function->getColorMap();
		TFColorMap::const_iterator first = points->begin();
		TFColorMap::const_iterator end = points->end();
		for(TFColorMap::const_iterator it = first; it != end; ++it)
		{
				writePoint_(*it);	
		}

	writeEndElement();
}

void TFXmlWriter::writePoint_(TFColor point){

	writeStartElement("TFColor");
		writeAttribute("component1", QString::fromStdString( convert<float, std::string>(point.component1)) );
		writeAttribute("component2", QString::fromStdString( convert<float, std::string>(point.component2)) );
		writeAttribute("component3", QString::fromStdString( convert<float, std::string>(point.component3)) );
		writeAttribute("alpha", QString::fromStdString( convert<float, std::string>(point.alpha)) );

	writeEndElement();
}

} // namespace GUI
} // namespace M4D
