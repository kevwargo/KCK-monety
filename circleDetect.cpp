/**
 * @brief Image enchancement and Hough transform example
 * @author Eugene Khvedchenya <ekhvedchenya@gmail.com>
 * @copyright computer-vision-talks.com/articles/how-to-detect-circles-in-noisy-image/
 */
 
 
 
#include <opencv2/opencv.hpp>
 
/**
 * These settings can be adjusted in runtime with OpenCV sliders.
 * Due to slider implementatio (lack of floating-point support) so values
 * are stored scaled by x10 or x100.
 */
struct EggsDetectorAlgorithmSettings
{
    int spatialWindowRadius;
    int colorWindowRadius;
    int sharpeningWeight; // x100
    int laplaccianScale; // x100
    int cannyThreshold;
    int accumulatorThreshold;
    int houghResolution; // x10
    int minRadius;
    int maxRadius;
    EggsDetectorAlgorithmSettings()
        : spatialWindowRadius(10)
        , colorWindowRadius(10)
        , sharpeningWeight(40)
        , laplaccianScale(20)
        , cannyThreshold(40)
        , accumulatorThreshold(40)
        , houghResolution(10)
        , minRadius(5)
        , maxRadius(20)
    {
    }
};
 
 
class EggsDetectorAlgorithm
{
public:
    void process(const cv::Mat_<cv::Vec3b>& inputRgbImage, const EggsDetectorAlgorithmSettings& settings)
    {
        cv::pyrMeanShiftFiltering(inputRgbImage, filtered, settings.spatialWindowRadius, settings.colorWindowRadius, 1);
        cv::cvtColor(filtered, grayImg, cv::COLOR_BGR2GRAY);
        grayImg.convertTo(grayImgf, CV_32F);
        cv::GaussianBlur(grayImgf, blurredf, cv::Size(5,5), 0);
        cv::Laplacian(blurredf, laplaccian, CV_32F);
        float weight = 0.01f * settings.sharpeningWeight;
        float scale = 0.01f * settings.laplaccianScale;
 
        cv::Mat_<float> sharpenedf = 1.5f * grayImgf
            - 0.5f * blurredf
            - weight * grayImgf.mul(scale * laplaccian);
        sharpenedf.convertTo(sharpened, CV_8U);
        // runs the actual detection
        cv::HoughCircles(sharpened,
                         circles,
                         cv::HOUGH_GRADIENT,
                         0.1f * settings.houghResolution,
                         settings.minRadius,
                         settings.cannyThreshold,
                         settings.accumulatorThreshold,
                         settings.minRadius,
                         settings.maxRadius );
    }
    cv::Mat display(const cv::Mat& colorImg) const
    {
        // clone the colour, input image for displaying purposes
        cv::Mat display = colorImg.clone();
        for( size_t i = 0; i < circles.size(); i++ )
        {
            cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            // circle center
            cv::circle( display, center, 3, cv::Scalar(0,255,0), -1, 8, 0 );
            // circle outline
            cv::circle( display, center, radius, cv::Scalar(0,0,255), 3, 8, 0 );
        }
 
        return display;
    }
private:
    cv::Mat filtered;
    cv::Mat grayImg;
    cv::Mat grayImgf;
    cv::Mat_<float> blurredf;
    cv::Mat_<float> laplaccian;
    cv::Mat sharpened;
    std::vector<cv::Vec3f> circles;
};
 
static const char * windowName = "Hough Circle Detection Demo";
 
class EggsDetectorBind
{
public:
    EggsDetectorBind(const cv::Mat_<cv::Vec3b>& inputRgbImage)
        : mImage(inputRgbImage)
    {
    }
    void run()
    {
 
        // create the main window, and attach the trackbars
        cv::namedWindow( windowName, cv::WINDOW_AUTOSIZE );
        cv::createTrackbar("Spatial window radius", windowName, &mSettings.spatialWindowRadius, 50, trackbarPropertyChanged, this);
        cv::createTrackbar("Color window radius", windowName, &mSettings.colorWindowRadius, 50, trackbarPropertyChanged, this);
        cv::createTrackbar("Sharpening weight", windowName, &mSettings.sharpeningWeight, 100, trackbarPropertyChanged, this);
        cv::createTrackbar("Laplaccian scale", windowName, &mSettings.laplaccianScale, 100, trackbarPropertyChanged, this);
        cv::createTrackbar("Canny threshold", windowName, &mSettings.cannyThreshold, 255, trackbarPropertyChanged, this);
        cv::createTrackbar("Accumulator Threshold", windowName, &mSettings.accumulatorThreshold, 100, trackbarPropertyChanged, this);
 
        cv::createTrackbar("Hough resolution", windowName, &mSettings.houghResolution, 100, trackbarPropertyChanged, this);
        cv::createTrackbar("Min Radius", windowName, &mSettings.minRadius, 50, trackbarPropertyChanged, this);
        cv::createTrackbar("Max Radius", windowName, &mSettings.maxRadius, 100, trackbarPropertyChanged, this);
        display();
 
        // infinite loop to display
        // and refresh the content of the output image
        // until the user presses q or Q
        int key = 0;
        while(key != 'q' && key != 'Q')
        {
            // those paramaters cannot be =0 so we must check here
            mSettings.spatialWindowRadius = std::max(mSettings.spatialWindowRadius, 1);
            mSettings.colorWindowRadius = std::max(mSettings.colorWindowRadius, 1);
 
            mSettings.cannyThreshold = std::max(mSettings.cannyThreshold, 1);
            mSettings.accumulatorThreshold = std::max(mSettings.accumulatorThreshold, 1);
 
            mSettings.houghResolution = std::max(mSettings.houghResolution, 1);
            mSettings.minRadius = std::max(mSettings.minRadius, 1);
            mSettings.maxRadius = std::max(mSettings.maxRadius, 1);
 
            // get user key
            key = cv::waitKey(10);
        }
    }
protected:
    void display()
    {
        mAlgorithm.process(mImage, mSettings);
        cv::imshow(windowName, mAlgorithm.display(mImage));
    }
 
    static void trackbarPropertyChanged(int, void* userdata)
    {
        EggsDetectorBind * self = (EggsDetectorBind *)userdata;
        self->display();
    }
private:
    cv::Mat mImage;
    EggsDetectorAlgorithmSettings mSettings;
    EggsDetectorAlgorithm mAlgorithm;
};
 
void main(int argc, const char * argv[])
{
    cv::Mat colorImg = cv::imread(argv[1]);
    while (colorImg.cols > 1000)
    {
        cv::Mat small;
        cv::pyrDown(colorImg, small);
        colorImg = small;
    }
    EggsDetectorBind bind(colorImg);
    bind.run();
} 
