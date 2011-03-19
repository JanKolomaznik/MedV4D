#include "TFXmlReader.h"

#include <TFHSVaFunction.h>
#include <TFRGBaFunction.h>

namespace M4D {
namespace GUI {

TFXmlReader::TFXmlReader() {}

TFXmlReader::~TFXmlReader(){}

void TFXmlReader::read(QIODevice* device, TFAbstractFunction::Ptr function, bool &error){
/*
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
	}*/
}

void TFXmlReader::readTestData(TFAbstractFunction* function){
	/*
	TF::ColorMapPtr f = function->getColorMap();
	TF::Size domain = function->getDomain();
	for(TF::Size i = 0; i < domain; ++i)
	{
		(*f)[i].component1 = (i/4)/1000.0;
		(*f)[i].component2 = (i/4)/1000.0;
		(*f)[i].component3 = (i/4)/1000.0;
		(*f)[i].alpha = (i/4)/1000.0;
	}*/
}

void TFXmlReader::readFunction_(TFAbstractFunction::Ptr function, bool &error){
	/*
	TF::Types::Function tfType = TF::convert<std::string, TF::Types::Function>(attributes().value("functionType").toString().toStdString());
	TF::Size domain = TF::convert<std::string, TF::Size>(attributes().value("domain").toString().toStdString());
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

	TF::ColorMapPtr points = function->getColorMap();
	TF::Size counter = 0;
	while (!atEnd())
	{
		readNext();

		if(isEndElement())
		{
			break;
		}

		if (isStartElement() && (name() == "TF::Color"))
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

void TFXmlReader::readPoint_(TF::Color* point, bool &error){
/*
	point->component1 = TF::convert<std::string,float>( attributes().value("component1").toString().toStdString() );
	point->component2 = TF::convert<std::string,float>( attributes().value("component2").toString().toStdString() );
	point->component3 = TF::convert<std::string,float>( attributes().value("component3").toString().toStdString() );
	point->alpha = TF::convert<std::string,float>( attributes().value("alpha").toString().toStdString() );

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
