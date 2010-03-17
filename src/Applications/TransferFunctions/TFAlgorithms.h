#ifndef TF_ALGORITHMS
#define TF_ALGORITHMS

#include <QtCore/QString>
#include <QtCore/QFile>

#include <QtGui/QWidget>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>
#include <QtCore/QXmlStreamReader>
#include <QtCore/QString>

#include <fstream>
#include <cassert>

#include <TFSimpleHolder.h>


class TFHolderFactory{

	class TFTypeSwitcher: public QXmlStreamReader{

	public:
		TFTypeSwitcher(){}
		~TFTypeSwitcher(){}

		TFType read(QIODevice* device){

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
					return convert<std::string, TFType>(attributes().value("type").toString().toStdString());
				}
			}
			return TFTYPE_UNKNOWN;
		}
	};

public:
	static TFAbstractHolder* load(QWidget* parent){
		
		QString fileName = QFileDialog::getOpenFileName(parent,
			QObject::tr("Open Transfer Function"),
			QDir::currentPath(),
			QObject::tr("TF Files (*.tf *.xml)"));

		if (fileName.isEmpty()) return NULL;

		QFile qFile(fileName);

		if (!qFile.open(QFile::ReadOnly | QFile::Text)) {
			QMessageBox::warning(parent, QObject::tr("Transfer Functions"),
							  QObject::tr("Cannot read file %1:\n%2.")
							  .arg(fileName)
							  .arg(qFile.errorString()));
			return NULL;
		}

		TFTypeSwitcher switcher;
		TFType tfType = switcher.read(&qFile);
		qFile.close();

		qFile.open(QFile::ReadOnly | QFile::Text);
		TFAbstractHolder* loaded = NULL;
		switch(tfType)
		{
			case TFTYPE_SIMPLE:
			{
				loaded = new TFSimpleHolder();
				if(!loaded->_load(qFile))
				{ 
					QMessageBox::warning(parent,
						QObject::tr("TFXmlSimpleReader"),
						QObject::tr("Parse error in file %1").arg(fileName));
				}
				break;
			}
			case TFTYPE_UNKNOWN:
			default:
			{
				assert("unknown holder");
				break;
			}
		}
		qFile.close();
		return loaded;
	}

private:
	TFHolderFactory(){}
	~TFHolderFactory(){}
};

template<typename Iterator>
void adjustByTransferFunction(
	typename std::iterator_traits<Iterator>::pointer pixel,
	typename std::iterator_traits<Iterator>::value_type min,
	typename std::iterator_traits<Iterator>::value_type max,
	const uint32 length,
	TFAbstractFunction *transferFunction){

		switch(transferFunction->getType()){
		case TFTYPE_SIMPLE:
		{
			adjustBySimpleFunction<Iterator>(pixel, min, max, length, transferFunction);
			break;
		}
		case TFTYPE_UNKNOWN:
		default:
		{
			assert("Unknown Transfer Function");
			break;
		}
	}
}

template<typename Iterator>
void adjustBySimpleFunction(
	typename std::iterator_traits<Iterator>::pointer pixel,
	typename std::iterator_traits<Iterator>::value_type min,
	typename std::iterator_traits<Iterator>::value_type max,
	const uint32 length,
	TFAbstractFunction *transferFunction){

	if ( !pixel || !transferFunction)
	{
		return;
	}

	TFSimpleFunction *tf = dynamic_cast<TFSimpleFunction*>(transferFunction);

	std::map<int, typename std::iterator_traits<Iterator>::value_type> computed;
	TFPoints points = tf->getAllPoints();

	if(points.empty())
	{
		return;
	}

	double range = max - min;

	TFPointsIterator first = points.begin();
	TFPointsIterator end = points.end();
	TFPointsIterator it = first;

	for (unsigned i = 0; i < length; ++i )
	{
		typename std::iterator_traits<Iterator>::value_type pixelValue = pixel[i];

		std::map<int, typename std::iterator_traits<Iterator>::value_type>::iterator stored = computed.find(pixelValue);
		if(stored != computed.end())
		{
			pixel[i] = stored->second;
			continue;
		}

		it = first;
		TFPoint lesser = *(it++);
		for(it; it != end; ++it)
		{
			if((*it).x/(double)FUNCTION_RANGE_SIMPLE > pixelValue/range)
			{
				break;
			}
			lesser = *it;
		}
		double lesserValue, greaterValue;
		double distance;
		if(it == end)
		{
			greaterValue = min;
		}
		else
		{
			greaterValue = (*it).y;
		}
		
		lesserValue = (lesser.y/(double)COLOR_RANGE_SIMPLE)*range;
		distance = ((pixelValue/range)*FUNCTION_RANGE_SIMPLE - lesser.x)/FUNCTION_RANGE_SIMPLE;
		typename std::iterator_traits<Iterator>::value_type result =
			(typename std::iterator_traits<Iterator>::value_type) ROUND( ((greaterValue - lesserValue)*distance) + lesserValue );

		computed.insert(std::make_pair(pixelValue, result));

		pixel[i] = result;
	}
}

#endif //TF_ALGORITHMS