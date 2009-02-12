
#include "itkGDCMImageIO.h"
#include "itkTIFFImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageSeriesWriter.h"

#include "segmentator.h"

int main( int argc, char* argv[] )
{

  if( argc < 9 || argc > 10)
    {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0];
    std::cerr << " DicomDirectory  outputFormat(tiff,dcm)";
    std::cerr << " seedX seedY seedZ ";
    std::cerr << " InitialDistance";
    std::cerr << " LowerThreshold";
    std::cerr << " UpperThreshold";
    std::cerr << " [CurvatureScaling == 1.0]" << std::endl;
    return EXIT_FAILURE;
    }  

  // create reader object
  typedef itk::ImageSeriesReader< InternalImageType >        ReaderType;
  ReaderType::Pointer reader = ReaderType::New();

  // create IO object that can handle DICOM
  typedef itk::GDCMImageIO       ImageIOType;
  ImageIOType::Pointer dicomIO = ImageIOType::New();
 
  reader->SetImageIO( dicomIO );

  // reading DICOM issues
  typedef itk::GDCMSeriesFileNames NamesGeneratorType;
  NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();

  nameGenerator->SetUseSeriesDetails( true );
  nameGenerator->AddSeriesRestriction("0008|0021" );

  nameGenerator->SetDirectory( argv[1] );

  typedef std::vector< std::string >    SeriesIdContainer;
  
  try
    {
    // read the specified directory end find out what DICOM series are within
	// note: there should be only ONE !!
    const SeriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();
    if(seriesUID.size() > 1)
    {
    	std::cerr << "There should be only ine serie within specified directory !!";
    	return EXIT_FAILURE;
    }
    
    std::cout << std::endl << "The directory: " <<  argv[1]  << std::endl;
    std::cout << "contains the following DICOM Series: ";
   
    SeriesIdContainer::const_iterator seriesItr = seriesUID.begin();
    SeriesIdContainer::const_iterator seriesEnd = seriesUID.end();
    while( seriesItr != seriesEnd )
      {
      std::cout << seriesItr->c_str() << std::endl;
      seriesItr++;
      }

    typedef std::vector< std::string >   FileNamesContainer;
    FileNamesContainer fileNames;

    // get the file names of files that compose the serie
    fileNames = nameGenerator->GetFileNames( seriesUID[0].c_str() );
    // and set them to reader
    reader->SetFileNames( fileNames );
    
    ////////////////////////////////////////////// result writer preparation ///////////////////////////////////    
    typedef itk::Image< ReadWritePixelType, 2 >         Image2DType;
    
    // create writer
    typedef itk::ImageSeriesWriter< ReadWriteImageType, Image2DType > WriterType;
    WriterType::Pointer writer = WriterType::New();
    
    std::cout  << "Writing the images " << std::endl;
    
    FileNamesContainer outFiles;
    std::stringstream strm;  
    
    std::string wantedFormat = argv[2];
    if(wantedFormat == "dcm")
    {
    	typedef itk::TIFFImageIO       OutImageIOType; 
    	OutImageIOType::Pointer outIO = OutImageIOType::New();
    	writer->SetImageIO(outIO);
    	
    	// prepare filenames for output images "<num>.dcm"    	      
	    for(unsigned int i=0; i<fileNames.size(); i++)
	    {
	    	strm << i << ".dcm";
	    	outFiles.push_back( strm.str() );
	    	strm.str("");
	    }
    } else if (wantedFormat == "tiff") {
    	typedef itk::TIFFImageIO       OutImageIOType;
    	OutImageIOType::Pointer outIO = OutImageIOType::New();
    	writer->SetImageIO(outIO);
    	    	
    	// prepare filenames for output images "<num>.tiff"    	      
	    for(unsigned int i=0; i<fileNames.size(); i++)
	    {
	    	strm << i << ".tiff";
	    	outFiles.push_back( strm.str() );
	    	strm.str("");
	    }
	} else {
		std::cerr << "unsuported  wanted format" << std::endl;
		return EXIT_FAILURE;
	}     
    
    writer->SetFileNames( outFiles);
    
    ////////////////////////////////////////////// actual work //////////////////////////////////////
    
    // update the reader to load the image
    reader->Update();
    
    float curvatureScaling = 1.0f;
    
    if(argc == 10)
    	curvatureScaling = atof(argv[9]);
    
    // create segmentation part of pipeline
    Segmentator segmentator(
    		atof(argv[8]), atof(argv[7]), curvatureScaling, 
    		atof(argv[3]), atof(argv[4]), atof(argv[5]), (double) atof(argv[6]),
    		reader->GetOutput()->GetBufferedRegion().GetSize());
    
    // connect it to readers and writers
    segmentator.thresholdSegmentation->SetFeatureImage( reader->GetOutput() );
    writer->SetInput( segmentator.thresholder->GetOutput() );
    
    // update the writer and thus invoke whole pipeline
    writer->Update();

    }
  catch (itk::ExceptionObject &ex)
    {
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
} 