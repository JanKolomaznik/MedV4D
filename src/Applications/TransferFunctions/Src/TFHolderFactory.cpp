#include<TFHolderFactory.h>


TFActions TFHolderFactory::createMenuTFActions(QWidget *owner, QMenu *menu){

	TFActions actions;

	//adding transferfunction types to menu
	actions.push_back(new TFAction(owner, menu, TFTYPE_SIMPLE));	
	//actions.push_back(new TFAction(owner, menu, TFTYPE_MYTYPE));

	return actions;
}

TFAbstractHolder* TFHolderFactory::createHolder(TFType &holderType){

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

TFAbstractHolder* TFHolderFactory::loadHolder(QWidget* parent){
	
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
	TFAbstractHolder* loaded = createHolder(tfType);

	if(loaded)
	{
		if(!loaded->load_(qFile))
		{ 
			QMessageBox::warning(parent,
				QObject::tr("TFXmlSimpleReader"),
				QObject::tr("Parse error in file %1").arg(fileName));
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