#ifndef TF_WINDOW
#define TF_WINDOW

#include <map>

#include <QtGui/QWidget>
#include <QtGui/QDockWidget>
#include <QtGui/QMainWindow>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>
#include <QtGui/QMenuBar>
#include <QtGui/QKeySequence>
#include <QtGui/QKeyEvent>
#include <QtGui/QVBoxLayout>

#include <TFCreator.h>
#include <TFPaletteButton.h>

#include <TFColor.h>
#include <TFHistogram.h>
#include <TFHolderInterface.h>

#include "ui_TFPalette.h"

namespace M4D {
namespace GUI {

class TFPalette : public QMainWindow{

    Q_OBJECT

public:

	TFPalette(QMainWindow* parent);
    ~TFPalette();

	//void setupDefault();

	void setDomain(const TF::Size domain);
	TF::Size getDomain();
	
	M4D::Common::TimeStamp getTimeStamp(/*bool& noFunctionAvailable*/);

	//TF::MultiDColor<dim>::Map::Ptr getColorMap();

	bool setHistogram(TF::Histogram::Ptr histogram, bool adjustDomain = true);

	template<typename BufferIterator>
	bool applyTransferFunction(BufferIterator begin, BufferIterator end){

		if(activeHolder_ < 0) on_actionNew_triggered();
		if(activeHolder_ < 0) exit(0);

		return palette_.find(activeHolder_)->second->applyTransferFunction<BufferIterator>(begin, end);
	}

private slots:

    void close_triggered(TF::Size index);

    void on_actionLoad_triggered();
	void on_actionNew_triggered();

	void change_activeHolder(TF::Size index);

protected:

	void resizeEvent(QResizeEvent*);

private:	

	typedef std::map<TF::Size, TFHolderInterface*> HolderMap;
	typedef HolderMap::iterator HolderMapIt;

	class Indexer{

		typedef std::vector<TF::Size> Indexes;
		typedef Indexes::iterator IndexesIt;

	public:

		Indexer();
		~Indexer();

		TF::Size getIndex();
		void releaseIndex(const TF::Size index);

	private:

		TF::Size nextIndex_;
		Indexes released_;
	};

    Ui::TFPalette* ui_;
	QMainWindow* mainWindow_;
	QVBoxLayout* layout_;

	TF::Histogram::Ptr histogram_;
	TF::Size domain_;

	M4D::Common::TimeStamp lastChange_;
	bool activeChanged_;

	Indexer indexer_;
	int activeHolder_;
	HolderMap palette_;

	void addToPalette_(TFHolderInterface* holder);
	void removeFromPalette_(const TF::Size index);
};

} // namespace GUI
} // namespace M4D

#endif //TF_WINDOW