#include <iostream>
#include <vector>
#include <string>

#include <boost/program_options.hpp>

// DLoopDetector and DBoW2
#include "DLoopDetector.h"
#include "StereoParameters.h"
#include <DBoW2/DBoW2.h>
#include <DVision/DVision.h>
#include "ORBextractor.h" // ORB_SLAM3 ORB Implementation

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "demoDetectorStereo.h"

using namespace DLoopDetector;
using namespace DBoW2;
using namespace DVision;
using namespace std;
namespace po = boost::program_options;

// TODO: remove dependency on these constants if possible
static const int IMAGE_W = 1280; // image size
static const int IMAGE_H = 720;

// FAST + BRIEF Keypoint Detector and Descriptor Configuration
static const int FAST_THRESH = 20;  // corner detector response threshold
static const bool FAST_NMS = true;  // perform non-maxima-suppression
static const int FAST_RETAIN_BEST = 2000;  //retain the best keypoints only
static const char *BRIEF_PATTERN_FILE = "./resources/brief_pattern.yml";
static const char *BRIEF_VOC_FILE = "./resources/brief_k10L6.voc.gz";

// ORB Keypoint Detector and Descriptor Configuration
static const char *ORB_VOC_FILE = "../resources/ORBvoc.yml.gz";
static const int ORB_NFEATURES = 2000;
static const float ORB_SCALEFACTOR = 1.2f;
static const int ORB_NLEVELS = 8;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// This functor extracts BRIEF descriptors in the required format
class BriefExtractor: public FeatureExtractor<BRIEF256::bitset>
{
public:
  /** 
   * Extracts features from an image
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  void operator()(const cv::Mat &im, 
    vector<cv::KeyPoint> &keys, vector<BRIEF256::bitset> &descriptors) const override;

  /**
   * Creates the brief extractor with the given pattern file
   * @param pattern_file
   */
  BriefExtractor(const std::string &pattern_file);

private:

  /// BRIEF descriptor extractor
  BRIEF256 m_brief;
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class OrbExtractor: public FeatureExtractor<cv::Mat>
{
public:
  /** 
   * Extracts features from an image
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  void operator()(const cv::Mat &im, 
    vector<cv::KeyPoint> &keys, vector<cv::Mat> &descriptors) const override;

  /**
   * Creates the brief extractor with the given pattern file
   * @param pattern_file
   */
  OrbExtractor(int nfeatures, float scalefactor, int nlevels);

private:

  // ORB descriptor and detector
  ORB_SLAM3::ORBextractor *orbext;
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

int main(int argc, char* argv[])
{
    std::string path_left;
    std::string path_right;
    std::string path_calibration;
    std::string poses_path;
    std::string type, voc_path;
    // Program Options (BOOST library w/special compilation needed)
    po::options_description options_desc(
        "StereoLoopDetector demo usage with ORB or BRIEF features.\n\n"
        "Allowed options");
    options_desc.add_options()
        ("help,h", "produce help message")
        ("input-left,l", po::value(&path_left)->required(),
            "path to file with newline separated paths to left camera input images")
        ("input-right,r", po::value(&path_right)->required(),
            "path to file with newline separated paths to right camera input images")
        ("calibration", po::value(&path_calibration)->required(), 
            "path to file with calibration parameters")
        ("poses-file", po::value(&poses_path)->required(), 
            "path to file with CSV poses x,y")
        ("type", po::value(&type)->default_value("ORB"), "Type of keypoint extractor/descriptor")
        ("voc", po::value(&voc_path), "Path to vocabulary file for specified type")
        ("noshow", po::value<bool>()->default_value(true), "Don't display results")
    ;

    // Parse program options
	po::variables_map options_vm;
	try {
		po::store(po::parse_command_line(argc, argv, options_desc), options_vm);
		if (options_vm.count("help")) {
			std::cout << options_desc << std::endl;
			return 0;
		}
		po::notify(options_vm);
	} catch (const po::error &ex) {
		std::cerr << ex.what() << std::endl;
		std::cout << std::endl;
		std::cout << options_desc << std::endl;
		return 1;
	}

    // Store options
    bool show = !options_vm["noshow"].as<bool>();

    // Validate options
    if (!((type == "BRIEF") || (type == "ORB"))) {
        std::cerr << "\"type\" option should be one of BRIEF or ORB, not " <<
        type << std::endl;
        throw po::validation_error(po::validation_error::invalid_option_value, "type");
    }

    StereoParameters sparams(path_calibration);

    // prepares the demo
    try 
    {
        std::string vocabulary;
        if (type == "BRIEF") {
        vocabulary = options_vm.count("voc")? voc_path : BRIEF_VOC_FILE;
        demoDetectorStereo<BriefVocabulary, BriefLoopDetector, FBrief::TDescriptor>
            demo(vocabulary, path_left, path_right, poses_path,
                 sparams, show);
        BriefExtractor extractor(BRIEF_PATTERN_FILE);
        demo.run(type, extractor);
        } else if (type == "ORB") {
        vocabulary = options_vm.count("voc")? voc_path : ORB_VOC_FILE;
        demoDetectorStereo<OrbVocabulary, OrbLoopDetector, FORB::TDescriptor>
            demo(vocabulary, path_left, path_right, poses_path,
                 sparams, show);
        OrbExtractor extractor(ORB_NFEATURES, ORB_SCALEFACTOR, ORB_NLEVELS);
        demo.run(type, extractor);
        }
    }
    catch(const std::string &ex)
    {
        cout << "Error: " << ex << endl;
    }

    return 0;
    }


    // ----------------------------------------------------------------------------

BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
    // The DVision::BRIEF extractor computes a random pattern by default when
    // the object is created.
    // We load the pattern that we used to build the vocabulary, to make
    // the descriptors compatible with the predefined vocabulary
    
    // loads the pattern
    cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
    if(!fs.isOpened()) throw string("Could not open file ") + pattern_file;
    
    vector<int> x1, y1, x2, y2;
    fs["x1"] >> x1;
    fs["x2"] >> x2;
    fs["y1"] >> y1;
    fs["y2"] >> y2;
    
    m_brief.importPairs(x1, y1, x2, y2);
}

// ----------------------------------------------------------------------------

void BriefExtractor::operator() (const cv::Mat &im, 
  vector<cv::KeyPoint> &keys, vector<BRIEF256::bitset> &descriptors) const
{
    // extract FAST keypoints with opencv
    cv::FAST(im, keys, FAST_THRESH, FAST_NMS);
    cv::KeyPointsFilter::retainBest(keys, FAST_RETAIN_BEST);
    
    // compute their BRIEF descriptor
    m_brief.compute(im, keys, descriptors);
}

// ----------------------------------------------------------------------------

OrbExtractor::OrbExtractor(int nfeatures, float scalefactor, int nlevels)
{
    std::cout << "Creating ORB extractor with nfeatures=" << nfeatures <<
        ", scalefactor=" << scalefactor << ", and nlevels=" << nlevels <<
        std::endl;
    orbext = new ORB_SLAM3::ORBextractor(nfeatures, scalefactor, nlevels, 20, 7);
}

// ----------------------------------------------------------------------------

void OrbExtractor::operator() (
  const cv::Mat &im, vector<cv::KeyPoint> &keys, 
  vector<cv::Mat> &descriptors) const
{
    cv::Mat descs;
    vector<int> vLapping = {0,0}; //Overlapping Pixels (stereo?) FIXME
    (*orbext)(im, cv::Mat(), keys, descs, vLapping);
    descriptors.resize(descs.rows);
    for(int i = 0; i < descs.rows; ++i)
    {
        descriptors[i] = descs.row(i);
    }
}

// ----------------------------------------------------------------------------
