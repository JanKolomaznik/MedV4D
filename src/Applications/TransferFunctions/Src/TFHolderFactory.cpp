#include<TFHolderFactory.h>


TFActions TFHolderFactory::createMenuTFActions(QWidget *owner, QMenu *menu){

	TFActions actions;

	actions.push_back(new TFAction(owner, menu, TFTYPE_SIMPLE));

	return actions;
}

TFAbstractHolder* TFHolderFactory::create(TFType &holderType){

	switch(holderType)
	{
		case TFTYPE_SIMPLE:
		{
			return new TFSimpleHolder();
		}
		case TFTYPE_UNKNOWN:
		default:
		{
			assert("unknown holder");
			break;
		}
	}
	return NULL;
}

TFAbstractHolder* TFHolderFactory::load(QWidget* parent){
	
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
			if(!loaded->load_(qFile))
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

TFType TFHolderFactory::TFTypeSwitcher::read(QIODevice* device){

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