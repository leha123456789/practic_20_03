#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
using namespace std;
using namespace cv;
void applyGrayscaleFilter(Mat& img)
{
    cvtColor(img, img, COLOR_BGR2GRAY);
}
void applySepiaFilter(Mat& img)
{
    Mat kernel = (Mat_<float>(3, 3) << 0.272, 0.534, 0.131, 0.349, 0.686, 0.168, 0.393, 0.769, 0.189);
    transform(img, img, kernel);
}
void applyNegativeFilter(Mat& img)
{
    bitwise_not(img, img);
}
void applySobelFilter(Mat& img)
{
    Mat sobelX, sobelY;
    Sobel(img, sobelX, CV_16S, 1, 0);
    Sobel(img, sobelY, CV_16S, 0, 1);
    Mat absSobelX, absSobelY;
    convertScaleAbs(sobelX, absSobelX);
    convertScaleAbs(sobelY, absSobelY);
    addWeighted(absSobelX, 0.5, absSobelY, 0.5, 0, img);
}
int main()
{
    Mat image = imread("input.jpg");
    Mat grayscaleImage = image.clone();
    Mat sepiaImage = image.clone();
    Mat negativeImage = image.clone();
    Mat edgeImage = image.clone();
#pragma omp parallel sections
    {
#pragma omp section
        {
            applyGrayscaleFilter(grayscaleImage);
            imshow("Серый", grayscaleImage);
            waitKey(0);
        }
#pragma omp section
        {
            applySepiaFilter(sepiaImage);
            imshow("Сепия", sepiaImage);
            waitKey(0);
        }
#pragma omp section
        {
            applyNegativeFilter(negativeImage);
            imshow("Негатив", negativeImage);
            waitKey(0);
        }
#pragma omp section
        {
            applyGrayscaleFilter(edgeImage);
            applySobelFilter(edgeImage);
            imshow("Контур", edgeImage);
            waitKey(0);
        }
    }
    return 0;
}
