#include "TFXmlReader.h"

#include <TFHSVaFunction.h>
#include <TFRGBaFunction.h>

namespace M4D {
namespace GUI {

TFXmlReader::TFXmlReader() {}

TFXmlReader::~TFXmlReader(){}

void TFXmlReader::read(QIODevice* device, TFAbstractFunction::Ptr function, bool &error){

	setDevice(device);

	while(!atEnd())
	{
		readNext(); 

		if(isEndElement())
		{
			break;
		}

		if (isStartElement() && (name() == "TransferFunction"))
		{
			readFunction_(function, error);
		}
	}
}

void TFXmlReader::readTestData(TFAbstractFunction* function){
	
	TFColorMapPtr f = function->getColorMap();
	TFSize domain = function->getDomain();
	for(TFSize i = 0; i < domain; ++i)
	{
		(*f)[i].component1 = (i/4)/1000.0;
		(*f)[i].component2 = (i/4)/1000.0;
		(*f)[i].component3 = (i/4)/1000.0;
		(*f)[i].alpha = (i/4)/1000.0;
	}
}

void TFXmlReader::readFunction_(TFAbstractFunction::Ptr function, bool &error){
	/*
	TFFunctionType tfType = convert<std::string, TFFunctionType>(attributes().value("functionType").toString().toStdString());
	TFSize domain = convert<std::string, TFSize>(attributes().value("domain").toString().toStdString());
	if(function && domain != function->getDomain())
	{
		delete function;
		function = NULL;
	}
	if(!function)
	{
		switch(tfType)
		{
			case TFFUNCTION_RGBA:
			{
				function = new TFRGBaFunction(domain);
				break;
			}
			case TFFUNCTION_HSVA:
			{
				function = new TFHSVaFunction(domain);
				break;
			}
			default:
			{
				error = true;
				return;
			}
		}
	}

	TFColorMapPtr points = function->getColorMap();
	TFSize counter = 0;
	while (!atEnd())
	{
		readNext();

		if(isEndElement())
		{
			break;
		}

		if (isStartElement() && (name() == "TFColor"))
		{
			if(counter >= domain)
			{
				error = true;
				break;
			}

			readPoint_(&((*points)[counter]), error);
			++counter;

			if(error)
			{
				break;
			}
		}
	}*/
}

void TFXmlReader::readPoint_(TFColor* point, bool &error){
/*
	point->component1 = convert<std::string,float>( attributes().value("component1").toString().toStdString() );
	point->component2 = convert<std::string,float>( attributes().value("component2").toString().toStdString() );
	point->component3 = convert<std::string,float>( attributes().value("component3").toString().toStdString() );
	point->alpha = convert<std::string,float>( attributes().value("alpha").toString().toStdString() );

	while (!atEnd())
	{
		readNext();
		
		if(isEndElement())
		{
			break;
		}
	}
	error = atEnd();*/
}

} // namespace GUI
} // namespace M4D
