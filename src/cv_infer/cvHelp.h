#pragma once
#include<iostream>
#include <numeric>
#include<opencv2/opencv.hpp>
#ifdef _WIN32
#include<io.h>
#else
#include<unistd.h>
#include<sys/stat.h>
#endif
#include<math.h>

#define PI       3.14159265358979323846
using namespace cv;
using namespace std;

struct OutputSeg {
	int id;             //������id
	float confidence;   //������Ŷ�
	cv::Rect box;       //���ο�
	cv::Mat boxMask;       //���ο���mask����ʡ�ڴ�ռ�ͼӿ��ٶ�
};

//�ַ���ת�ǶȲ���
struct OcrAngleParams
{
	int ocrCenterX;//�ַ���������λ��x
	int ocrCenterY;//�ַ���������λ��y
	float angle;//������ַ���б�Ƕ�
	int class_id;//������
	Point2f p0;//��С��Ӿ��ε��ĸ���������//Сͼ����
	Point2f p1;//��С��Ӿ��ε��ĸ���������
	Point2f p2;//��С��Ӿ��ε��ĸ���������
	Point2f p3;//��С��Ӿ��ε��ĸ���������
	/*Point2f P_top[4];*/
	Point2i left_top;//���������
	float height;//���ߵĳ���

};

struct RotatedRect_my
{
	int CenterX;//�ַ���������λ��x
	int CenterY;//�ַ���������λ��y
	float angle;//��б�Ƕ�
	Point2f p0;//��С��Ӿ��ε��ĸ���������//Сͼ����
	Point2f p1;//��С��Ӿ��ε��ĸ���������
	Point2f p2;//��С��Ӿ��ε��ĸ���������
	Point2f p3;//��С��Ӿ��ε��ĸ���������
	float height;//���ߵĳ���
};
struct pointd {
	double x;
	double y;
};

//��ȡ��Ч������ַ����
struct OcrRoiResult
{
	int CenterX;//�ַ���������λ��x//��ԭͼ������
	int CenterY;//�ַ���������λ��y
	cv::Mat ocrPicture;//�ַ�Ƭ�ε�ͼ��
};

struct OcrRecognitionResult
{
	int CenterX;//�ַ���������λ��x//��ԭͼ������
	int CenterY;//�ַ���������λ��y
	string zifu;//�ַ����
	int cls_label;//�Ƕȷ����ǩ
};

struct recDataRead
{
	int stationNumber;//��λ��
	string imagePath;//ͼ��·��
	//string delimiter;//�ָ����//ռλʹ��
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
//ͼ��ƽ��
Mat imgTranslate(Mat& matSrc, int xOffset, int yOffset, bool bScale);

OcrAngleParams Minrect(Mat img_th);

//���ճ��ȴ�С������ð�ݷ�������
void bubbleSort_height(vector<OcrAngleParams>& nums);

// �����pΧ�Ƶ�center��ʱ����תangle�Ⱥ��������
Point2f rotatePoint(const Point2f& p, const Point2f& center, float angle);

void rotatePoint2(double angle, pointd& rotate_pt, pointd origin_pt, pointd center_pt);
