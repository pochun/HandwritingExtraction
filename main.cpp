#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

int main(int argc, const char* argv[])
{
    fs::path inputPath;
    fs::path outputPath;
    
    po::options_description desc("Extract handwriting from lecture videos");
    try {
        desc.add_options()
            ("help,h", "Help screen")
            ("input,i", po::value<fs::path>()->required(), "Input video")
            ("output,o", po::value<fs::path>()->required(), "Output video");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        
        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }
        
        po::notify(vm);
        
        inputPath = vm["input"].as<fs::path>();
        outputPath = vm["output"].as<fs::path>();
        
    } catch (const po::error& ex) {
        std::cerr << ex.what() << std::endl;
        std::cout << desc << std::endl;
        return 1;
    }
    
    cv::VideoCapture cap(inputPath.string());
    if(!cap.isOpened())
        return -1;

    const int resolution = 480;

    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);

    const double scale = (double)resolution / height;

    height = resolution;
    width = (int)(width * scale);

    cv::VideoWriter output(
        outputPath.string(),
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        30,
        cv::Size(width * 2, height));

    const int windowSize = 20;
    std::list<cv::Mat> frameWindow;
    std::list<cv::Mat> outputWindow;

    int frameCount = 0;
    cv::Vec3f colorBoard = cv::Vec3f(0.f);
    cv::Mat lastMask;
    while (true)
    {
        cv::Mat frame;
        cap >> frame; // get a new frame from camera
        
        if (frame.empty())
            break;

        // cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        cv::resize(frame, frame, cv::Size(width, height));

#if 0
        cv::Mat ycrcb;
        cv::cvtColor(frame, ycrcb, cv::COLOR_BGR2YCrCb);
        std::vector<cv::Mat> channels;
        cv::split(ycrcb, channels);

        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(4);

        clahe->apply(channels[0], channels[0]);
        cv::merge(channels, ycrcb);

        cv::cvtColor(ycrcb, frame, cv::COLOR_YCrCb2BGR);
#endif

        cv::Mat data, frame32f;

        frame.convertTo(frame32f, CV_32F);

        const int sz = 64;

        if (frameCount < 30)
        {
            data = frame32f.reshape(1, frame32f.total());

            cv::Mat labels, centers;
            cv::kmeans(data, 5, labels, cv::TermCriteria(cv::TermCriteria::MAX_ITER, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

            centers = centers.reshape(3, centers.rows);

            std::vector<int> count(centers.rows, 0);
            for (int i = 0; i < labels.rows; ++i)
                ++count[labels.at<int>(i, 0)];

            int maxLabel = std::max_element(count.begin(), count.end()) - count.begin();
            cv::Mat draw(sz, sz, centers.type(), cv::Scalar::all(0));
            colorBoard *= frameCount;
            colorBoard += centers.at<cv::Vec3f>(maxLabel, 0);
            colorBoard /= (frameCount + 1);
            draw = colorBoard;
            draw.convertTo(draw, CV_8U);
            //cv::cvtColor(draw, draw, cv::COLOR_YCrCb2BGR);
            // cv::imshow("centers", draw);
        }

        cv::Mat intensity = cv::Mat(frame32f.size(), CV_32FC1);

        for (int i = 0; i < intensity.rows; ++i)
            for (int j = 0; j < intensity.cols; ++j) {
                cv::Vec3f c = frame32f.at<cv::Vec3f>(i, j) - colorBoard;
                intensity.at<float>(i, j) = std::sqrt(c.dot(c));
            }

        cv::Mat intensity8u;
        intensity.convertTo(intensity8u, CV_8U);
        cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

        cv::Mat intensityOpened;
        cv::morphologyEx(intensity8u, intensityOpened, cv::MORPH_OPEN, element);

        cv::Mat subtraction;
        cv::subtract(intensity8u, intensityOpened, subtraction);

        cv::Mat binarized;
        cv::threshold(subtraction, binarized, 0., 255., cv::THRESH_OTSU);

        cv::Mat lowIntensity;
        cv::threshold(intensity8u, lowIntensity, 80., 255., cv::THRESH_BINARY_INV);

        if (frameWindow.size() < windowSize) {
            frameWindow.push_back(intensity);
        } else {
            frameWindow.pop_front();
            frameWindow.push_back(intensity);
        }

        cv::Mat sum = cv::Mat::zeros(intensity.size(), intensity.type());
        cv::Mat sqsum = cv::Mat::zeros(intensity.size(), intensity.type());

        for (auto it = frameWindow.begin(); it != frameWindow.end(); ++it) {
            cv::accumulate(*it, sum);
            cv::accumulateSquare(*it, sqsum);
        }

        sum /= frameWindow.size();
        
        cv::multiply(sum, sum, sum);
        sqsum /= frameWindow.size();

        cv::Mat highVariance;
        cv::subtract(sqsum, sum, highVariance);
        highVariance.convertTo(highVariance, CV_8U);

        cv::threshold(highVariance, highVariance, 0., 255., cv::THRESH_OTSU);

        cv::Mat outputFrame, lowVariance;
        cv::bitwise_not(highVariance, lowVariance);
        if (lastMask.empty()) {
            // cv::bitwise_and(lowVariance, binarized, output);
            outputFrame = cv::Mat::zeros(binarized.size(), binarized.type());
        }
        else
        {
            cv::Mat updateMask, updateMask2;

            //cv::morphologyEx(lowVariance, lowVariance, cv::MORPH_CLOSE, element);
            cv::erode(lowVariance, lowVariance, element);
            //cv::morphologyEx(lowIntensity, lowIntensity, cv::MORPH_CLOSE, element);
            cv::erode(lowIntensity, lowIntensity, element);

            cv::bitwise_xor(lastMask, binarized, updateMask);
            cv::bitwise_and(updateMask, lowVariance, updateMask);
            cv::bitwise_or(binarized, lowIntensity, updateMask2);
            cv::bitwise_and(updateMask, updateMask2, updateMask);
            cv::bitwise_xor(lastMask, updateMask, outputFrame);
        }

        lastMask = outputFrame;
        //cv::imshow("frame", frame);
        //cv::imshow("output", output);

        cv::Mat display;
        cv::cvtColor(outputFrame, outputFrame, cv::COLOR_GRAY2BGR);
        cv::hconcat(frame, outputFrame, display);

        // cv::imshow("result", display);

        output.write(display);

        //int c = cv::waitKey(1);
        //if (c == 27)
        //    break;

        ++frameCount;
    }

    cap.release();
    output.release();

    // std::cout << "mIOU: " << miou << std::endl;
    return 0;
}
