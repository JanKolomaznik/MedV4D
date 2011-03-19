#include "TFXmlWriter.h"

#include <TFRGBaFunction.h>

namespace M4D {
namespace GUI {

TFXmlWriter::TFXmlWriter(){
	setAutoFormatting(true);
}

void TFXmlWriter::write(QIODevice *device, TFAbstractFunction::Ptr data){
/*
	setDevice(device);

	writeStartDocument();
		writeDTD("<!DOCTYPE TransferFunctions>");

		writeFunction_(data);

	writeEndDocument();*/
}

void TFXmlWriter::writeTestData(QIODevice *device){
	/*
	TFAbstractFunction::Ptr data(new TFRGBaFunction(3000));
	TF::ColorMapPtr f = data->getColorMap();
	TF::Size domain = data->getDomain();
	for(TF::Size i = 0; i < domain; ++i)
	{
		(*f)[i].component1 = (i/4)/1000.0;
		(*f)[i].component2 = (i/4)/1000.0;
		(*f)[i].component3 = (i/4)/1000.0;
		(*f)[i].alpha = (i/4)/1000.0;
	}
	write(device, data);	*/
}

void TFXmlWriter::writeFunction_(TFAbstractFunction::Ptr function){
	/*
	writeStartElement("TransferFunction");
		//writeAttribute("holderType", QString::fromStdString(TF::convert<TF::Types::Predefined, std::string>(holderType)));
		writeAttribute("functionType", QString::fromStdString(TF::convert<TF::Types::Function, std::string>(function->getType())));
		writeAttribute("domain", QString::fromStdString( TF::convert<TF::Size, std::string>(function->getDomain()) ));

		const TF::ColorMapPtr points = function->getColorMap();
		TF::ColorMap::const_iterator first = points->begin();
		TF::ColorMap::const_iterator end = points->end();
		for(TF::ColorMap::const_iterator it = first; it != end; ++it)
		{
				writePoint_(*it);	
		}

	writeEndElement();*/
}

void TFXmlWriter::writePoint_(TF::Color point){
/*
	writeStartElement("TF::Color");
		writeAttribute("component1", QString::fromStdString( TF::convert<float, std::string>(point.component1)) );
		writeAttribute("component2", QString::fromStdString( TF::convert<float, std::string>(point.component2)) );
		writeAttribute("component3", QString::fromStdString( TF::convert<float, std::string>(point.component3)) );
		writeAttribute("alpha", QString::fromStdString( TF::convert<float, std::string>(point.alpha)) );

	writeEndElement();*/
}

} // namespace GUI
} // namespace M4D
