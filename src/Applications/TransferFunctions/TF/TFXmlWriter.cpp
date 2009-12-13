#include "TFXmlWriter.h"

 TFXmlWriter::TFXmlWriter(){
     setAutoFormatting(true);
 }

 bool TFXmlWriter::write(QIODevice *device, TFScheme** data){

     setDevice(device);

     writeStartDocument();
     writeDTD("<!DOCTYPE TFScheme>");

	 writeScheme(data);
	
     writeEndDocument();
     
	 return true;
 }

 void TFXmlWriter::writeScheme(TFScheme** scheme){

     writeStartElement("TFScheme");
	 writeAttribute("name", QString::fromStdString((*scheme)->name));

	vector<TFName> points = (*scheme)->getFunctionNames();
	vector<TFName>::iterator first = points.begin();
	vector<TFName>::iterator end = points.end();
	vector<TFName>::iterator it = first;

     for(it; it != end; ++it)
	 {
		 TFFunction* f = (*scheme)->getFunction(*it);
		 writeFunction(&f);
		 delete f;
	 }
         
	 writeEndElement();
 }

 void TFXmlWriter::writeFunction(TFFunction** function){

     writeStartElement("TFFunction");
	 writeAttribute("name", QString::fromStdString((*function)->name));
     writeAttribute("colourR", QString::fromStdString( convert<int, string>((*function)->colourRGB[0]) ));
     writeAttribute("colourG", QString::fromStdString( convert<int, string>((*function)->colourRGB[1]) ));
     writeAttribute("colourB", QString::fromStdString( convert<int, string>((*function)->colourRGB[2]) ));

	vector<TFPoint*> points = (*function)->getAllPoints();
	vector<TFPoint*>::iterator first = points.begin();
	vector<TFPoint*>::iterator end = points.end();
	vector<TFPoint*>::iterator it = first;

     for(it; it != end; ++it)
	 {
         writePoint(&(*it));
		 delete *it;
	 }

     writeEndElement();
 }

 void TFXmlWriter::writePoint(TFPoint** point){

     writeStartElement("TFPoint");
     writeAttribute("x", QString::fromStdString( convert<int, string>((*point)->x)) );
     writeAttribute("y", QString::fromStdString( convert<int, string>((*point)->y)) );

     writeEndElement();	//jednorazovy element?
 }