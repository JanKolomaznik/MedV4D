#ifndef COLOR_CHOOSER_BUTTON_H
#define COLOR_CHOOSER_BUTTON_H

#include <QtGui>

class ColorChooserButton: public QPushButton
{
	Q_OBJECT;
public:
	ColorChooserButton( QWidget *parent = NULL ): QPushButton( parent ), mColor( 255,255,255,255 ), mAllowAlphaSetting( false )
	{
		mColorDialog = new QColorDialog( this );
		QObject::connect( this, SIGNAL( clicked() ), this, SLOT( showDialog() ) );
		QObject::connect( mColorDialog, SIGNAL( colorSelected( const QColor & ) ), this, SLOT( setColor( const QColor & ) ) );
	}
	QColor
	color()const
	{
		return mColor;
	}

	bool
	isAlphaEnabled()const
	{
		return mAllowAlphaSetting;
	}
public slots:
	void
	setColor( const QColor &aColor ) {
		mColor = aColor;
		update();
	}

	void
	enableAlpha( bool aEnable )
	{
		mAllowAlphaSetting = aEnable;
	}
signals:

protected slots:
	void
	showDialog()
	{
		mColorDialog->setOption( QColorDialog::ShowAlphaChannel, mAllowAlphaSetting );
		mColorDialog->setCurrentColor( mColor );
		mColorDialog->open();
		//mColor = mColorDialog->currentColor();
		update();
	}

protected:
	void
	paintEvent ( QPaintEvent *event )
	{
		//QPushButton::paintEvent ( event );

		QPainter painter(this);
		QRect rect = QRect( QPoint(), size() );
		//TODO	
		painter.setBrush( QBrush( mColor ) );
		painter.drawRect(rect);
	}


	QColor mColor;
	bool mAllowAlphaSetting;
	QColorDialog *mColorDialog;

};


#endif /*COLOR_CHOOSER_BUTTON_H*/
