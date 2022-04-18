#include <iostream>

#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace aruco;

void drawCube(Mat frame, vector <Point2f> corners, const Vec3d& rvec, const Vec3d& tvec,
              const Mat& camMatrix, const Mat& distCoeffs);


static bool readDetectorParameters(const string& filename, aruco::DetectorParameters &params) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["adaptiveThreshWinSizeMin"] >> params.adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params.adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params.adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params.adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params.minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params.maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params.polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params.minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params.minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params.minMarkerDistanceRate;
    fs["cornerRefinementMethod"] >> params.cornerRefinementMethod;
    fs["cornerRefinementWinSize"] >> params.cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params.cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params.cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params.markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params.perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params.perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params.maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params.minOtsuStdDev;
    fs["errorCorrectionRate"] >> params.errorCorrectionRate;

    return true;
}


int main() {

    DetectorParameters params;

    readDetectorParameters("../detector_params.yml", params);

    Dictionary dict = getPredefinedDictionary(PREDEFINED_DICTIONARY_NAME(10));


    double c[] = {1567.7598663928327, 0, 959.50000000000000, 0, 1569.8260146860121, 539.50000000000000, 0, 0,1};
    Mat camMatrix = {
            Size(3, 3),
            CV_64F,
            c
    };

    double d[] =     {1.7746202293461785e-2, -6.7204665235718858e-2, 0, 0, 3.5079887214058908e-1};

    Mat distCoeffs = {Size(5, 1),
                      CV_64F,
                      d};



    vector<int> ids;
    vector<vector<Point2f>> corners, rejected;
    vector<Vec3d> rvecs, tvecs;

    const Ptr<Dictionary> dic = &dict;
    const Ptr<DetectorParameters>par = &params;


    VideoCapture cap("../2.mp4");
    while (true){


        Mat frame;
        cap >> frame;
        if(frame.empty()){
            break;
        }

//        imshow("Frame", frame);

        detectMarkers(frame, dic,  corners, ids, par, rejected, camMatrix, distCoeffs);

        if(!ids.empty()){
            estimatePoseSingleMarkers(corners, 1, camMatrix, distCoeffs, rvecs, tvecs);
            for(auto i = 0; i < ids.size(); i++){
                drawCube(frame, corners[i], rvecs[i], tvecs[i], camMatrix, distCoeffs);
            }

        }


        imshow("Frame", frame);




        waitKey(0);

    }


}

void drawCube(Mat frame, vector <Point2f> corners, const Vec3d& rvec, const Vec3d& tvec,
              const Mat& camMatrix, const Mat& distCoeffs){

    vector<Point3d> source = {
            {-0.5, 0.5, 1},
            {0.5, 0.5, 1},
            {0.5, -0.5, 1},
            {-0.5, -0.5, 1},
    };

    vector <Point2d> upperCorners;

    projectPoints(source, rvec, tvec, camMatrix, distCoeffs, upperCorners);

        for (auto i:{0, 1, 2, 3}) {
        line(frame, upperCorners[i], corners[i] , Scalar(255, 12, 12), 3);
    }

    for (auto i: {0, 1, 2}){
        line(frame, upperCorners[i], upperCorners[i + 1], Scalar(0, 0, 255), 3);
        line(frame, corners[i], corners[i + 1], Scalar(0, 156, 0), 3);
    }

    line(frame, upperCorners[3], upperCorners[0], Scalar(0, 0, 255), 3);
    line(frame, corners[3], corners[0], Scalar(0, 156, 0), 3);

}


