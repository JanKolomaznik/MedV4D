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

#include "ui_TFPalette.h"

namespace M4D {
namespace GUI {

class TFPalette : public QMainWindow{

    Q_OBJECT

	typedef std::map<TF::Size, TFHolder*> HolderMap;
	typedef HolderMap::iterator HolderMapIt;

public:

	TFPalette(QMainWindow* parent);
    ~TFPalette();

	//void setupDefault();

	void setDomain(TF::Size domain);
	TF::Size getDomain();
	
	M4D::Common::TimeStamp getTimeStamp(bool& noFunctionAvailable);

	template<typename BufferIterator>
	bool applyTransferFunction(BufferIterator begin, BufferIterator end){

		if(activeHolder_ < 0) on_actionNew_triggered();

		return palette_.find(activeHolder_)->second->applyTransferFunction<BufferIterator>(begin, end);
	}

	template<typename HistogramIterator>
	bool setHistogram(HistogramIterator begin, HistogramIterator end, bool adjustDomain = true){

		histogram_ = TF::Adaptation::computeTFHistogram<HistogramIterator>(begin, end);
		
		if(adjustDomain) domain_ = histogram_->size();
		else if(domain_ != histogram_->size()) return false;

		HolderMapIt beginPalette = palette_.begin();
		HolderMapIt endPalette = palette_.end();
		for(HolderMapIt it = beginPalette; it != endPalette; ++it)
		{
			it->second->setHistogram(histogram_);
		}
		return true;
	}

private slots:

    void close_triggered(TF::Size index);

    void on_actionLoad_triggered();
	void on_actionNew_triggered();

	void change_activeHolder(TF::Size index);

protected:

	void resizeEvent(QResizeEvent*);

private:	

	class Indexer{

		typedef std::vector<TF::Size> Indexes;
		typedef Indexes::iterator IndexesIt;

	public:

		Indexer();
		~Indexer();

		TF::Size getIndex();
		void releaseIndex(TF::Size index);

	private:

		TF::Size nextIndex_;
		Indexes released_;
	};

    Ui::TFPalette* ui_;
	QMainWindow* mainWindow_;
	QVBoxLayout* layout_;

	TF::Histogram::Ptr histogram_;
	TF::Size domain_;

	Indexer indexer_;
	int activeHolder_;
	HolderMap palette_;

	void addToPalette_(TFHolder* holder);
	void removeFromPalette_(TF::Size index);
};

} // namespace GUI
} // namespace M4D

#endif //TF_WINDOW