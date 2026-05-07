#include"cvHelp.h"


double CalculateAngle(Point Mar1Point, Point Mar2Point)
{
	double xx, yy;
	double arcLength1 = 0;
	double current_angle = 0;
	xx = Mar2Point.x - Mar1Point.x;
	yy = Mar2Point.y - Mar1Point.y;
	if (xx == 0.0)
	{
		arcLength1 = PI / 2.0;
		current_angle = 90;
	}
	else
	{
		double k = yy / xx;  //计算斜率
		arcLength1 = atan(k);    //弧度
		current_angle = arcLength1 * 180 / PI;  //角度
	}
	return current_angle;
}
bool cmp0(const RotatedRect_my& a, const RotatedRect_my& b)
{
	return a.height > b.height;  //x升序排列 +
}

OcrAngleParams Minrect(Mat img_th)
{
	//Mat img_th;
	//cvtColor(img, img_th, COLOR_BGR2GRAY);
	//threshold(img_th, img_th, 0, 255, THRESH_OTSU);//自适应二值化
												   //寻找最外层轮廓  
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(img_th, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());

	Mat img_hb = Mat::zeros(img_th.size(), CV_8UC1); //最小外接矩形画布 

	vector<RotatedRect_my>rect_vec;
	for (int i = 0; i < contours.size(); i++)
	{
		RotatedRect_my middle0;

		//绘制轮廓  
		drawContours(img_hb, contours, i, Scalar(255), 1, 8, hierarchy);

		//绘制轮廓的最小外接矩形  
		RotatedRect rect = minAreaRect(contours[i]);
		Point2f P[4];

		rect.points(P);
		for (int j = 0; j <= 3; j++)
		{
			/*line(img_th, P[j], P[(j + 1) % 4], Scalar(0, 0, 255), 25);
			std::cout << "x=" << P[j].x << " y=" << P[j].y << std::endl;*/
		}
		middle0.p0 = P[0];
		middle0.p1 = P[1];
		middle0.p2 = P[2];
		middle0.p3 = P[3];

		double distance_w = sqrt(pow((P[0].x - P[1].x), 2) + pow(P[0].y - P[1].y, 2));
		double distance_h = sqrt(pow((P[1].x - P[2].x), 2) + pow(P[1].y - P[2].y, 2));
		/*cout << "distance_w==" << distance_w << endl;
		cout << "distance_h==" << distance_h << endl;*/
		double angle_rec;
		if (distance_w > distance_h)
		{
			angle_rec = CalculateAngle(P[0], P[1]);
			rect.size.height = distance_w;
		}
		else
		{
			angle_rec = CalculateAngle(P[1], P[2]);
			rect.size.height = distance_h;
		}
		rect.angle = angle_rec;

		middle0.angle = angle_rec;
		middle0.CenterX = rect.center.x;
		middle0.CenterY = rect.center.y;
		middle0.height = rect.size.height;
		rect_vec.push_back(middle0);
	}
	sort(rect_vec.begin(), rect_vec.end(), cmp0);
	//cout << rect_vec[0].angle << endl;
	float angle = rect_vec[0].angle;
	/*cout << "rect_vec[0].center.x=" << rect_vec[0].center.x << endl;
	cout << "rect_vec[0].center.y=" << rect_vec[0].center.y << endl;*/
	OcrAngleParams oap;
	oap.angle = angle;
	oap.ocrCenterX = rect_vec[0].CenterX;
	oap.ocrCenterY = rect_vec[0].CenterY;
	oap.class_id = 0;
	/*Point2f P2[4];
	rect_vec[0].points(P2);*/
	oap.p0 = rect_vec[0].p0;
	oap.p1 = rect_vec[0].p1;
	oap.p2 = rect_vec[0].p2;
	oap.p3 = rect_vec[0].p3;
	oap.left_top = Point2i(0, 0);
	oap.height = rect_vec[0].height;

	/*for (int i=0;i<4;i++)
	{*/
	//std::cout << "111x=" << rect_vec[0].p0.x << " 111y=" << rect_vec[0].p0.y << std::endl;
	/*}*/

	/*cv::namedWindow("最小外接矩形", 0);
	cv::imshow("最小外接矩形", img_th);
	cv::waitKey();*/
	return oap;
}
//图像平移
Mat imgTranslate(Mat& matSrc, int xOffset, int yOffset, bool bScale)
{
	// 判断是否改变图像大小,并设定被复制ROI
	int nRows = matSrc.rows;
	int nCols = matSrc.cols;
	int nRowsRet = 0;
	int nColsRet = 0;
	Rect rectSrc;
	Rect rectRet;
	if (bScale)
	{
		nRowsRet = nRows + abs(yOffset);
		nColsRet = nCols + abs(xOffset);
		rectSrc.x = 0;
		rectSrc.y = 0;
		rectSrc.width = nCols;
		rectSrc.height = nRows;
	}
	else
	{
		nRowsRet = matSrc.rows;
		nColsRet = matSrc.cols;
		if (xOffset >= 0)
		{
			rectSrc.x = 0;
		}
		else
		{
			rectSrc.x = abs(xOffset);
		}
		if (yOffset >= 0)
		{
			rectSrc.y = 0;
		}
		else
		{
			rectSrc.y = abs(yOffset);
		}
		rectSrc.width = nCols - abs(xOffset);
		rectSrc.height = nRows - abs(yOffset);
	}
	// 修正输出的ROI
	if (xOffset >= 0)
	{
		rectRet.x = xOffset;
	}
	else
	{
		rectRet.x = 0;
	}
	if (yOffset >= 0)
	{
		rectRet.y = yOffset;
	}
	else
	{
		rectRet.y = 0;
	}
	rectRet.width = rectSrc.width;
	rectRet.height = rectSrc.height;
	// 复制图像
	Mat matRet(nRowsRet, nColsRet, matSrc.type(), Scalar(0));
	matSrc(rectSrc).copyTo(matRet(rectRet));
	return matRet;
}

Mat Rotate(Mat img, int ocrCenterX, int ocrCenterY, float angle, int& xOffset, int& yOffset)
{
	/*cv::Mat matScale_0;
	int w = img.cols;
	int h = img.rows;*/

	//Point2f center(img.cols / 2, img.rows / 2);//中心
	Point2f center_ocr(ocrCenterX, ocrCenterY);//字符中心
	Point2f center_p(img.cols / 2, img.rows / 2);//图像中心
	xOffset = center_p.x - center_ocr.x;
	yOffset = center_p.y - center_ocr.y;

	// 平移图像
	Mat matScale_0 = imgTranslate(img, xOffset, yOffset, false);//平移图像//尺寸不变

	double angle0 = angle;
	double scale = 1;
	Mat roateM = getRotationMatrix2D(center_p, angle0, scale);  //获得旋转矩阵,顺时针为负，逆时针为正
	//// 下几行代码用于调整旋转后图像的位置，确保旋转后图像完全可见
	//double cos = abs(roateM.at<double>(0, 0));
	//double sin = abs(roateM.at<double>(0, 1));
	//// 旋转后图像的宽度nw和高度nh
	//int nw = int(abs(cos) * w + abs(sin) * h);
	//int nh = int(abs(sin) * w + abs(cos) * h);
	//// 旋转后图像的中心点位置
	//roateM.at<double>(0, 2) += (nw / 2 - w / 2);
	//roateM.at<double>(1, 2) += (nh / 2 - h / 2);
	//// 由于计算出的 nw 和 nh 可能是浮点数，但 cv::warpAffine()函数的第四个参数（目标图像的大小）需要整数类型
	//cv::Size newSize(nw, nh);
	//xOffset = (nw / 2 - w / 2);
	//xOffset = (nh / 2 - h / 2);
	//// 将图像按照定义的rotation_matrix旋转变换的矩阵信息，进行旋转
	//warpAffine(img, matScale_0, roateM, newSize); //仿射变换
	// 
	warpAffine(matScale_0, matScale_0, roateM, Size(matScale_0.cols, matScale_0.rows)); //仿射变换
	//// 定义图像截取范围
	//int startRow = 0;
	//int endRow = 2000;
	//int startCol = 0;
	//int endCol = 2000;
	//// 使用cv::Range截取图像范围
	//cv::Range rowRange(startRow, endRow);
	//cv::Range colRange(startCol, endCol);
	//cv::Mat roi = image(rowRange, colRange);
	return matScale_0;
}

Mat RotateOnly(Mat img, float angle)
{
	/*cv::Mat matScale_0;
	int w = img.cols;
	int h = img.rows;*/
	cv::Mat matScale_0;

	//Point2f center(img.cols / 2, img.rows / 2);//中心
	//Point2f center_ocr(ocrCenterX, ocrCenterY);//字符中心
	Point2f center_p(img.cols / 2, img.rows / 2);//图像中心
	//xOffset = center_p.x - center_ocr.x;
	//yOffset = center_p.y - center_ocr.y;

	//// 平移图像
	//Mat matScale_0 = imgTranslate(img, xOffset, yOffset, false);//平移图像//尺寸不变

	double angle0 = angle;
	double scale = 1;
	Mat roateM = getRotationMatrix2D(center_p, angle0, scale);  //获得旋转矩阵,顺时针为负，逆时针为正

	warpAffine(img, matScale_0, roateM, Size(img.cols, img.rows)); //仿射变换
	//// 定义图像截取范围
	//int startRow = 0;
	//int endRow = 2000;
	//int startCol = 0;
	//int endCol = 2000;
	//// 使用cv::Range截取图像范围
	//cv::Range rowRange(startRow, endRow);
	//cv::Range colRange(startCol, endCol);
	//cv::Mat roi = image(rowRange, colRange);
	return matScale_0;
}

void DrawPred(Mat& img, vector<OutputSeg> result, std::vector<std::string> classNames, vector<Scalar> color, bool isVideo) {
	Mat mask = img.clone();
	vector<OcrAngleParams> jiaozheng_result;
	for (int i = 0; i < result.size(); i++) {
		int left, top;
		left = result[i].box.x;
		top = result[i].box.y;
		int color_num = i;
		rectangle(img, result[i].box, color[result[i].id], 2, 8);
		int sum_x = 0;
		int sum_y = 0;
		int number_p = 0;
		if (result[i].boxMask.rows && result[i].boxMask.cols > 0)
		{
			Mat roi_mask = result[i].boxMask.clone(); //最小外接矩形画布 
			//Mat img_hb = Mat::zeros(result[i].boxMask.size(), CV_8UC1); //最小外接矩形画布 
			OcrAngleParams angle_result = Minrect(roi_mask);

			std::cout << "旋转角度" << angle_result.angle << std::endl;
			OcrAngleParams oap2;
			oap2.angle = angle_result.angle;
			oap2.ocrCenterX = angle_result.ocrCenterX + left;
			oap2.ocrCenterY = angle_result.ocrCenterY + top;
			oap2.class_id = result[i].id;
			jiaozheng_result.push_back(oap2);

			mask(result[i].box).setTo(color[result[i].id], result[i].boxMask);
			int dim = result[i].boxMask.channels();
			std::cout << "深度维数：" << dim << std::endl;
			if (dim == 1) {
				int pv = result[i].boxMask.at<uchar>(result[i].boxMask.rows / 2, result[i].boxMask.cols / 2);
				std::cout << "像素值：" << pv << std::endl;
			}
			//求取语义区域的质心
			for (int i1 = 0; i1 < result[i].boxMask.rows; i1++)
			{
				for (int j = 0; j < result[i].boxMask.cols; j++)
				{
					//uchar* p = result[i].boxMask.ptr<uchar>(i, j);
					if (result[i].boxMask.at<uchar>(i1, j) == 255)
					{
						sum_x = sum_x + j;
						sum_y = sum_y + i1;
						number_p++;
					}

				}
			}
			std::cout << "中心坐标:x==" << result[i].box.width / 2 << " y==" << result[i].box.height / 2 << std::endl;
			std::cout << "质心坐标:x==" << sum_x / number_p << " y==" << sum_y / number_p << std::endl;
		}

		string label = classNames[result[i].id] + ":" + to_string(result[i].confidence);
		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height);
		//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
		cv::putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
	}
	addWeighted(img, 0.5, mask, 0.5, 0, img); //add mask to src
	cv::namedWindow("result_mask", 0);
	cv::namedWindow("result_addWeight", 0);
	cv::imshow("result_mask", mask);
	cv::imshow("result_addWeight", img);
	if (!isVideo)
		waitKey();
	//destroyAllWindows();

}

//按照长度大小，进行冒泡法的排序
void bubbleSort_height(vector<OcrAngleParams>& nums) {
	int n = nums.size();
	bool swapped;

	for (int i = 0; i < n - 1; i++) {
		swapped = false;

		for (int j = 0; j < n - i - 1; j++) {
			if (nums[j].height > nums[j + 1].height) {
				swap(nums[j], nums[j + 1]);
				swapped = true;
			}
		}

		// 如果一轮遍历没有发生交换，说明序列已经有序，提前结束排序
		if (!swapped) {
			break;
		}
	}
}

// 计算点p围绕点center逆时针旋转angle度后的新坐标
Point2f rotatePoint(const Point2f& p, const Point2f& center, float angle) {
	float radians = angle * PI / 180.0; // 将角度转换为弧度
	float cosValue = std::cos(radians);
	float sinValue = std::sin(radians);

	// 应用旋转矩阵变换
	Point2f result;
	result.x = (p.x - center.x) * cosValue - (p.y - center.y) * sinValue + center.x;
	result.y = (p.x - center.x) * sinValue + (p.y - center.y) * cosValue + center.y;

	return result;
}

void rotatePoint2(double angle, pointd& rotate_pt, pointd origin_pt, pointd center_pt)
{
	double x0 = center_pt.x;
	double y0 = center_pt.y;

	double x = origin_pt.x;
	double y = origin_pt.y;

	rotate_pt.x = (x - x0) * cos(angle * PI / 180) - (y - y0) * sin(angle * PI / 180) + x0;
	rotate_pt.y = (x - x0) * sin(angle * PI / 180) + (y - y0) * cos(angle * PI / 180) + y0;
}