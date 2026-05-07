#pragma once
#include<iostream>
#include <numeric>
#include<opencv2/opencv.hpp>
#include<io.h>
#include<math.h>

#define PI       3.14159265358979323846
using namespace cv;
using namespace std;

struct OutputSeg {
	int id;             //结果类别id
	float confidence;   //结果置信度
	cv::Rect box;       //矩形框
	cv::Mat boxMask;       //矩形框内mask，节省内存空间和加快速度
};

//字符旋转角度参数
struct OcrAngleParams
{
	int ocrCenterX;//字符中心坐标位置x
	int ocrCenterY;//字符中心坐标位置y
	float angle;//计算的字符倾斜角度
	int class_id;//类别序号
	Point2f p0;//最小外接矩形的四个顶点坐标//小图坐标
	Point2f p1;//最小外接矩形的四个顶点坐标
	Point2f p2;//最小外接矩形的四个顶点坐标
	Point2f p3;//最小外接矩形的四个顶点坐标
	/*Point2f P_top[4];*/
	Point2i left_top;//左上坐标点
	float height;//长边的长度

};

struct RotatedRect_my
{
	int CenterX;//字符中心坐标位置x
	int CenterY;//字符中心坐标位置y
	float angle;//倾斜角度
	Point2f p0;//最小外接矩形的四个顶点坐标//小图坐标
	Point2f p1;//最小外接矩形的四个顶点坐标
	Point2f p2;//最小外接矩形的四个顶点坐标
	Point2f p3;//最小外接矩形的四个顶点坐标
	float height;//长边的长度
};
struct pointd {
	double x;
	double y;
};

//截取有效区域的字符结果
struct OcrRoiResult
{
	int CenterX;//字符中心坐标位置x//在原图的坐标
	int CenterY;//字符中心坐标位置y
	cv::Mat ocrPicture;//字符片段的图像
};

struct OcrRecognitionResult
{
	int CenterX;//字符中心坐标位置x//在原图的坐标
	int CenterY;//字符中心坐标位置y
	string zifu;//字符结果
	int cls_label;//角度分类标签
};

struct recDataRead
{
	int stationNumber;//工位号
	string imagePath;//图像路径
	//string delimiter;//分割符号//占位使用
};

struct zifu_center_info {
	cv::Point2f pt1;
	cv::Point2f pt2;
	float center_y;
};
double CalculateAngle(Point Mar1Point, Point Mar2Point);
bool cmp0(const RotatedRect_my& a, const RotatedRect_my& b);

Mat Rotate(Mat img, int ocrCenterX, int ocrCenterY, float angle, int& xOffset, int& yOffset);
Mat RotateOnly(Mat img, float angle);
//图像平移
Mat imgTranslate(Mat& matSrc, int xOffset, int yOffset, bool bScale);

OcrAngleParams Minrect(Mat img_th);

//按照长度大小，进行冒泡法的排序
void bubbleSort_height(vector<OcrAngleParams>& nums);

// 计算点p围绕点center逆时针旋转angle度后的新坐标
Point2f rotatePoint(const Point2f& p, const Point2f& center, float angle);

void rotatePoint2(double angle, pointd& rotate_pt, pointd origin_pt, pointd center_pt);
