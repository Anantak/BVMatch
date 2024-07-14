#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "bvftdescriptors.h"

using namespace cv;

int main(int argc, char** argv) 
{

  // Check arguments
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <image1> <image2>" << std::endl;
    return 1;
  }

  // Load images
  Mat img1 = imread(argv[1], IMREAD_GRAYSCALE);
  Mat img2 = imread(argv[2], IMREAD_GRAYSCALE);

  // Check if images loaded successfully
  if (img1.empty() || img2.empty()) {
    std::cerr << "Error loading images!" << std::endl;
    return 1;
  }

    BVFT bvfts1 = detectBVFT(img1);
    BVFT bvfts2 = detectBVFT(img2);

    float rows = img1.rows, cols = img1.cols;

    int max_x1 = img1.cols;
    int max_y1 = img1.rows;
    int max_x2 = img2.cols;
    int max_y2 = img2.rows;

    bvfts2.keypoints.insert(bvfts2.keypoints.end(),bvfts2.keypoints.begin(),bvfts2.keypoints.end());

    Mat temp(bvfts2.keypoints.size(),bvfts2.descriptors.cols,CV_32F,Scalar{0} );
    bvfts2.descriptors.copyTo(temp(Rect(0,0,bvfts2.descriptors.cols,bvfts2.keypoints.size()/2)));
    int areas = 6;
    int feautre_size=bvfts2.descriptors.cols/areas/areas;
    for(int i=0; i<areas*areas; i++)  //areas*areas
    {
        bvfts2.descriptors(Rect((areas*areas-i-1)*feautre_size,0 , feautre_size,bvfts2.keypoints.size()/2)).copyTo(temp(Rect(i*feautre_size, bvfts2.keypoints.size()/2, feautre_size , bvfts2.keypoints.size()/2))); //i*norient
    }
    bvfts2.descriptors = temp.clone();

    BFMatcher matcher;//(NORM_L2, true);
    vector<DMatch> matches;
    matcher.match(bvfts1.descriptors, bvfts2.descriptors, matches);

    vector<Point2f> points1;
    vector<Point2f> points2;
    vector<DMatch>::iterator it_end = matches.end();
    for(vector<DMatch>::iterator it= matches.begin(); it!= it_end;it++)
    {
        Point2f point_local_1 = (Point2f(max_y1,max_x1)- 
                                    Point2f(bvfts1.keypoints[it->queryIdx].pt.y,bvfts1.keypoints[it->queryIdx].pt.x))*0.4;
        points1.push_back(point_local_1);

        point_local_1 = (Point2f(max_y2,max_x2)- 
                                    Point2f(bvfts2.keypoints[it->trainIdx].pt.y,bvfts2.keypoints[it->trainIdx].pt.x))*0.4;
        points2.push_back(point_local_1);
    }
    cv::Mat keypoints1(points1);
    keypoints1=keypoints1.reshape(1,keypoints1.rows); //N*2
    cv::Mat keypoints2(points2);
    keypoints2=keypoints2.reshape(1,keypoints2.rows);

    // Mat inliers;
    // vector<int> inliers_ind;
    // Mat rigid = estimateICP(keypoints1, keypoints2, inliers_ind);
    // if (inliers_ind.size()<4) cout << "few inlier points" << endl;
    // cout << "find transform: \n" << rigid << endl;
    // //return;
    // vector<DMatch> good_matches;
    // for(int i=0; i<inliers_ind.size(); i++)
    //   good_matches.push_back(matches[inliers_ind[i]]);

    Mat matchesImage;

    drawKeypoints(img1, bvfts1.keypoints,img1,Scalar(0,0,255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(img2, bvfts2.keypoints,img2,Scalar(0,0,255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    drawMatches(img1, bvfts1.keypoints, img2, bvfts2.keypoints, matches, matchesImage, Scalar::all(-1), 
    // drawMatches(img1, bvfts1.keypoints, img2, bvfts2.keypoints, good_matches, matchesImage, Scalar::all(-1), 
		Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    imshow("matchesImage", matchesImage);  
    imwrite("match.png",matchesImage);
    waitKey(0);

  /*
  // Detect FAST corners
  Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
  detector->setThreshold(50);
  std::vector<KeyPoint> keypoints1, keypoints2;
  detector->detect(img1, keypoints1);
  detector->detect(img2, keypoints2);

  // Compute descriptors (using ORB for this example)
  Ptr<ORB> extractor = ORB::create();
  Mat descriptors1, descriptors2;
  extractor->compute(img1, keypoints1, descriptors1);
  extractor->compute(img2, keypoints2, descriptors2);

  // Match keypoints using Brute-Force matcher
  Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING);
  std::vector<DMatch> matches;
  matcher->match(descriptors1, descriptors2, matches);

  // Draw matches
  Mat result;
  drawMatches(img1, keypoints1, img2, keypoints2, matches, result);

  // Show result
  imshow("Matches", result);
  waitKey(0);
  */

  return 0;
}
