/*双目追踪的步骤：立体标定、立体校正、立体匹配、前景提取、目标追踪*/
#include "opencv2/opencv.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/video/background_segm.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace cv;
using namespace std;

Size imageSize;//图像的size
int nimages;
vector<string> goodImageList;//检测到的棋盘图像（因为有的棋盘图像是检测不到的）

static bool readStringList( const string& filename, vector<string>& l )//现在这里是读取每个图像,把图像名放到imagelist 容器中
{
	l.resize(0);
	FileStorage fs(filename, FileStorage::READ);
	if( !fs.isOpened() )//未打开则为false
		return false;
	FileNode n = fs.getFirstTopLevelNode();
	if( n.type() != FileNode::SEQ )
		return false;
	FileNodeIterator it = n.begin(), it_end = n.end();
	for( ; it != it_end; ++it )
		l.push_back((string)*it);//这里是把读取到的图像的名字放入容器
	return true;
}

static void StereoCalib(const vector<string>& imagelist, Size boardSize,  bool showRectified=true)
{
    if( imagelist.size() % 2 != 0 )//判断左右棋盘图像是否为偶数
    {
        cout << "ERROR: 采集的棋盘图像必须为偶数！\n";
        return;
    }

    bool displayCorners = false;//显示棋盘角点
    const int maxScale = 2;//缩放图像
    const float squareSize = 1.f;  // 棋盘格大小

    vector<vector<Point2f> > imagePoints[2];//这个是左右图像中的二维点
    vector<vector<Point3f> > objectPoints;//由上面的二维点得到的三维点

    int i, j, k;
	nimages = (int)imagelist.size()/2;//j是用于记录最后检测到了多少对棋盘图

    imagePoints[0].resize(nimages);//0为左图像；1为右图像---这里的左右是相对的
    imagePoints[1].resize(nimages);

    for( i = j = 0; i < nimages; i++ )
    {
        for( k = 0; k < 2; k++ )
        {
            const string& filename = imagelist[i*2+k];//这里是图像名
			string path="C:/Users/lenovo/Desktop/【重要】视觉程序/追踪程序/【完整】双目追踪/棋盘图/";//棋盘图像路径
			path.append(filename);//读取文件路径为：C:/Users/lenovo/Desktop/【重要】视觉程序/【完整】双目追踪/棋盘图/xx.jpg

            Mat img = imread(path, 0);
            if(img.empty())
                break;
            if( imageSize == Size() )
                imageSize = img.size();
            else if( img.size() != imageSize )
            {
                cout << "图像 " << filename << "与第一幅图像的尺寸不一样，这个时候就跳过这一对图像！\n";
                break;
            }

            bool found = false;//查找棋盘角点
            vector<Point2f>& corners = imagePoints[k][j];//左右边的第k幅盘图像
            for( int scale = 1; scale <= maxScale; scale++ )//看是否要对图像进行缩放
            {
                Mat timg;
                if( scale == 1 )
                    timg = img;
                else
                    resize(img, timg, Size(), scale, scale);
                found = findChessboardCorners(timg, boardSize, corners,
                    CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
                if( found )//有角点后的操作
                {
                    if( scale > 1 )
                    {
                        Mat cornersMat(corners);//放大后复原
                        cornersMat *= 1./scale;
                    }
                    break;
                }
            }
            if( displayCorners )//是否将画出角点的棋盘图显示出来，变量在第40行定义
            {
                cout << filename << endl;//哪副图-只显示名字
                Mat cimg, cimg1;
                cvtColor(img, cimg, COLOR_GRAY2BGR);
                drawChessboardCorners(cimg, boardSize, corners, found);//画角点
                double sf = 640./MAX(img.rows, img.cols);
                resize(cimg, cimg1, Size(), sf, sf);
                imshow("corners", cimg1);
                char c = (char)waitKey(500);
                if( c == 27 || c == 'q' || c == 'Q' )
                    exit(-1);
            }
            else
                putchar('.');//不显示的时候的操作
            if( !found )//没找到角点则退出
                break;
            cornerSubPix(img, corners, Size(11,11), Size(-1,-1),        //亚像素角点
                         TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,
                                      30, 0.01));
        }
        if( k == 2 )//内层循环完后将j加 1 ，并把找到的棋盘图放入容器
        {
            goodImageList.push_back(imagelist[i*2]);
            goodImageList.push_back(imagelist[i*2+1]);
            j++;
        }
    }
    cout << "\n共检测到"<<j << "对棋盘图。\n";
    nimages = j;
    if( nimages < 2 )
    {
        cout << "ERROR: 棋盘图太少，不足用于标定！\n";
        return;
    }

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    objectPoints.resize(nimages);

    for( i = 0; i < nimages; i++ )
    {
        for( j = 0; j < boardSize.height; j++ )
            for( k = 0; k < boardSize.width; k++ )
                objectPoints[i].push_back(Point3f(j*squareSize, k*squareSize, 0));//将二维点转为三维点
    }

    cout << "开始立体标定 ...\n";

    Mat cameraMatrix[2], distCoeffs[2];//内参数和畸变参数
    cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
    cameraMatrix[1] = Mat::eye(3, 3, CV_64F);
    Mat R, T, E, F;//旋转平移矩阵、本征矩阵、基础矩阵

    double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],   //立体标定，返回均方根误差rms
                    cameraMatrix[0], distCoeffs[0],
                    cameraMatrix[1], distCoeffs[1],
                    imageSize, R, T, E, F,
                    TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5),
                    CV_CALIB_FIX_ASPECT_RATIO +
                    CV_CALIB_ZERO_TANGENT_DIST +
                    CV_CALIB_SAME_FOCAL_LENGTH +
                    CV_CALIB_RATIONAL_MODEL +
                    CV_CALIB_FIX_K3 + CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5);
    cout << "均方根误差(RMS ERROR)：" << rms << endl;

    /* 验证标定的效果：由于输出的基础矩阵包含所有的输出信息，所以这里可以用对极几何约束（m2^t*F*m1=0）来验证-----下面这段程序可以不要*/
#if 0       //这里的0 改成1 就能运行下面这段程序了！
    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];
    for( i = 0; i < nimages; i++ )
    {
        int npt = (int)imagePoints[0][i].size();
        Mat imgpt[2];
        for( k = 0; k < 2; k++ )
        {
            imgpt[k] = Mat(imagePoints[k][i]);
            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);//返回未畸变的点
            computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);//计算对应的极线
        }
        for( j = 0; j < npt; j++ )
        {
            double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
                                imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
                           fabs(imagePoints[1][i][j].x*lines[0][j][0] +
                                imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    cout << "平均重投影误差：" <<  err/npoints << endl;
    /*到这里就验证完了*/
#endif

    FileStorage fs("C:/Users/lenovo/Desktop/【重要】视觉程序/追踪程序/【完整】双目追踪/棋盘图/intrinsics.yml", CV_STORAGE_WRITE);//保存内参数
    if( fs.isOpened() )
    {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
            "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
        fs.release();
    }
    else
        cout << "ERROR: 保存内参数失败！\n";

	fs.open("C:/Users/lenovo/Desktop/【重要】视觉程序/追踪程序/【完整】双目追踪/棋盘图/relative extrinsics.yml",CV_STORAGE_WRITE);
	if (fs.isOpened())
	{
		fs<< "R" << R << "T" << T;
		fs.release();
	}
	else
	cout<<"ERROR:保存两个相机的相对外参数失败！\n";
	cout<<"立体标定完毕！\n";
}

static vector<Mat> StereoRectified(Mat &img1,Mat &img2)//返回两个ROI
{
	Mat cameraMatrix[2], distCoeffs[2];//内参数和畸变参数
	Mat R, T;//两个相机的相对旋转和平移参数
	FileStorage fs("C:/Users/lenovo/Desktop/【重要】视觉程序/追踪程序/【完整】双目追踪/棋盘图/intrinsics.yml",CV_STORAGE_READ);
	if (fs.isOpened())
	{
		fs["M1"] >> cameraMatrix[0]; 
		fs["D1"] >> distCoeffs[0];
		fs["M2"] >> cameraMatrix[1];
		fs["D2"] >> distCoeffs[1];
		fs.release();
	}
	fs.open("C:/Users/lenovo/Desktop/【重要】视觉程序/追踪程序/【完整】双目追踪/棋盘图/relative extrinsics.yml",CV_STORAGE_READ);
	if (fs.isOpened())
	{
		fs["R"] >> R;
		fs["T"] >> T;
		fs.release();
	}
	
	Mat R1, R2, P1, P2, Q;
	Rect validRoi[2];//校正后的ROI
	vector<Rect>ROIs;//最后将两个ROI放进去，返回

	stereoRectify(cameraMatrix[0], distCoeffs[0],       //立体校正
		cameraMatrix[1], distCoeffs[1],
		imageSize, R, T, R1, R2, P1, P2, Q,
		CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);
	for(int i=0;i<2;i++)//把这两个Rect返回
	{
		ROIs.push_back(validRoi[i]);
	}

	fs.open("C:/Users/lenovo/Desktop/【重要】视觉程序/【完整】双目追踪/棋盘图/extrinsics.yml", CV_STORAGE_WRITE);//保存外参数
	if( fs.isOpened() )
	{
		fs  << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
		fs.release();
	}
	else
		cout << "ERROR:保存外参数失败！\n";

	// OpenCV 可以处理左右放置和上下放置的相机
	bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

	Mat rmap[2][2];//校正映射：左右图像各两个

	//计算映射矩阵
	initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	Mat canvas;//这个图像的长是棋盘图的两倍，高和棋盘图像一样（这是在水平放置的情况下）
	double sf;
	int w, h;
	if( !isVerticalStereo )//这个是水平放置的
	{
		sf = 600./MAX(imageSize.width, imageSize.height);
		w = cvRound(imageSize.width*sf);//取整
		h = cvRound(imageSize.height*sf);
		canvas.create(h, w*2, CV_8UC3);
	}
	else//这是上下放置的
	{
		sf = 300./MAX(imageSize.width, imageSize.height);
		w = cvRound(imageSize.width*sf);
		h = cvRound(imageSize.height*sf);
		canvas.create(h*2, w, CV_8UC3);
	}

#if 0 //这里是画出校正后的棋盘图：对棋盘图进行校正、画出校正后的可用ROI、画出左右两边对极后的极线
	//如果执行这里，则要等这里检测到的所有棋盘图都画完之后才会执行后续操作
	int i,j,k;
	for( i = 0; i < nimages; i++ )//这个是将一对图像画到一幅图像上
	{
		for( k = 0; k < 2; k++ )
		{
			const string &filename=goodImageList[i*2+k];
			string path = "C:/Users/lenovo/Desktop/【重要】视觉程序/追踪程序/【完整】双目追踪/棋盘图/";//棋盘图像路径
			path.append(filename);

			Mat img = imread(path, 0), rimg, cimg;
			remap(img, rimg, rmap[k][0], rmap[k][1], CV_INTER_LINEAR);//对图像原进行重映射，映射输出为校正后的图像
			cvtColor(rimg, cimg, COLOR_GRAY2BGR);
			Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));//这里是得出校正两个ROI
			resize(cimg, canvasPart, canvasPart.size(), 0, 0, CV_INTER_AREA);//缩放

			Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),           //得到两个矩形的坐标
				cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
			rectangle(canvasPart, vroi, Scalar(0,0,255), 3, 8); //将矩形画出来
		}

		if( !isVerticalStereo )//水平
			for( j = 0; j < canvas.rows; j += 16 )   //画极线
				line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
		else
			for( j = 0; j < canvas.cols; j += 16 )
				line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
		imshow("rectified", canvas);
		char c = (char)waitKey();
		if( c == 27 || c == 'q' || c == 'Q' )
			break;
	}
#endif

	//下面是对采集来的图像进行校正
	Mat  rimg1,rimg2;
	remap(img1, rimg1, rmap[0][0], rmap[0][1], CV_INTER_LINEAR);//对图像原进行重映射，映射输出为校正后的图像
	remap(img2, rimg2, rmap[1][0], rmap[1][1], CV_INTER_LINEAR);
	//imshow("校正左",rimg1);
	//imshow("校正右",rimg2);
	rectangle(rimg1,validRoi[0],Scalar(0,0,255),3,8);
	rectangle(rimg2,validRoi[1],Scalar(0,0,255),3,8);
	imshow("校正后左边的ROI",rimg1);
	imshow("校正后右边的ROI",rimg2);
	
	Mat rectMat[2];//裁剪校正后的ROI,裁剪之后就是校正后的，左右两边差异较小
	rectMat[0]=img1(validRoi[0]);
	rectMat[1]=img1(validRoi[1]);
	/*imshow("校正后裁剪左边的ROI",rectMat[0]);
	imshow("校正后裁剪右边的ROI",rectMat[1]);*/

	vector<Mat>recMat;//把裁剪后的图像放进来返回
	recMat.push_back(rectMat[0]);
	recMat.push_back(rectMat[1]);
	return recMat;
}

Mat element=getStructuringElement(MORPH_RECT,Size(7,7));//形态学滤波结构元
static Mat StereoMatch(Mat &img1,Mat &img2)
{
	Mat rectGray[2];//立体匹配是对灰度图进行的，所以这里转换为灰度图
	cvtColor(img1,rectGray[0],COLOR_BGR2GRAY);
	cvtColor(img2,rectGray[1],COLOR_BGR2GRAY);
	GaussianBlur(rectGray[0],rectGray[0],Size(3,3),0,0);
	GaussianBlur(rectGray[0],rectGray[0],Size(3,3),0,0);
	/*下面开始立体匹配：对裁剪后的图像进行立体匹配->后面恢复原图像用原图像的宽/高和裁剪后的图像的宽/高获得*/
	Mat disp,disp8;
#if 1
	StereoBM bm;
	bm.state->preFilterCap = 31;
	bm.state->SADWindowSize = 9;
	bm.state->minDisparity = 0;
	bm.state->numberOfDisparities = 16;
	bm.state->textureThreshold = 10;
	bm.state->uniquenessRatio = 15;
	bm.state->speckleWindowSize = 100;
	bm.state->speckleRange = 32;
	bm.state->disp12MaxDiff = 1;
	//bm(rectGray[1], rectGray[0], disp);//这里得到的视差图不是很好
	bm(rectGray[0], rectGray[1], disp);
	disp.convertTo(disp8, CV_8U, 255/(16*16.));
#else   //这个里面的两种算法耗时太长，采集到的图像实时性很差
	StereoSGBM sgbm;
	int cn = rectMat[0].channels();
	sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.minDisparity = 0;
	sgbm.numberOfDisparities = 16;
	sgbm.uniquenessRatio = 10;
	sgbm.speckleWindowSize = bm.state->speckleWindowSize;
	sgbm.speckleRange = bm.state->speckleRange;
	sgbm.disp12MaxDiff = 1;
	sgbm(rectGray[0], rectGray[1], disp);
	//sgbm(rectGray[1], rectGray[0], disp);
	disp.convertTo(disp8, CV_8U, 255/(16*16.));

	StereoVar var;
	var.levels = 3;                                 // ignored with USE_AUTO_PARAMS
	var.pyrScale = 0.5;                             // ignored with USE_AUTO_PARAMS
	var.nIt = 25;
	var.minDisp = -16;
	var.maxDisp = 0;
	var.poly_n = 3;
	var.poly_sigma = 0.0;
	var.fi = 15.0f;
	var.lambda = 0.03f;
	var.penalization = var.PENALIZATION_TICHONOV;   // ignored with USE_AUTO_PARAMS
	var.cycle = var.CYCLE_V;                        // ignored with USE_AUTO_PARAMS
	var.flags = var.USE_SMART_ID | var.USE_AUTO_PARAMS | var.USE_INITIAL_DISPARITY | var.USE_MEDIAN_FILTERING ;
	var(rectGray[0], rectGray[1], disp);
	//var(rectGray[1], rectGray[0], disp);
	disp.convertTo(disp8, CV_8U, 255/(16*16.));
#endif
	morphologyEx(disp8,disp8,MORPH_DILATE,element);//对视差图膨胀处理
	morphologyEx(disp8,disp8,MORPH_ERODE,element);//膨胀后对视差图腐蚀处理
	return disp8;
}
//下面是光流相关的函数和变量
#define UNKNOWN_FLOW_THRESH 1e9  

void makecolorwheel(vector<Scalar> &colorwheel) //这里相当于做一个画板 
{  
	int RY = 15;  
	int YG = 6;  
	int GC = 4;  
	int CB = 11;  
	int BM = 13;  
	int MR = 6;  

	int i;  

	/*把各种颜色丢进画板*/
	for (i = 0; i < RY; i++) colorwheel.push_back(Scalar(255,       255*i/RY,     0));  
	for (i = 0; i < YG; i++) colorwheel.push_back(Scalar(255-255*i/YG, 255,       0));  
	for (i = 0; i < GC; i++) colorwheel.push_back(Scalar(0,         255,      255*i/GC));  
	for (i = 0; i < CB; i++) colorwheel.push_back(Scalar(0,         255-255*i/CB, 255));  
	for (i = 0; i < BM; i++) colorwheel.push_back(Scalar(255*i/BM,      0,        255));  
	for (i = 0; i < MR; i++) colorwheel.push_back(Scalar(255,       0,        255-255*i/MR));  
}  

void motionToColor(Mat flow, Mat &color)  
{  
	if (color.empty()) //判断是否有数据
		color.create(flow.rows, flow.cols, CV_8UC3);  

	static vector<Scalar> colorwheel; //这个容器里有r,g,b三种颜色 
	if (colorwheel.empty())         //做画板
		makecolorwheel(colorwheel);  

	float maxrad = -1;  //确定运动范围，相当于运动半径

	/*查找最大光流值（根号（fx^2+fy^2）的最大值）用于将fx和fy归一化*/  
	for (int i= 0; i < flow.rows; ++i)   
	{  
		for (int j = 0; j < flow.cols; ++j)   
		{  
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);  //坐标为（i,j）的点的光流值
			float fx = flow_at_point[0];  //光流的两个分量
			float fy = flow_at_point[1];  
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))  //光流分量分别和设定的阈值比较
				continue;  
			float rad = sqrt(fx * fx + fy * fy);  //用于确定运动范围
			maxrad = maxrad > rad ? maxrad : rad;  //取最大值
		}  
	}  

	for (int i= 0; i < flow.rows; ++i)   
	{  
		for (int j = 0; j < flow.cols; ++j)   
		{  
			uchar *data = color.data + color.step[0] * i + color.step[1] * j;  //访问光流图的像素值
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);  

			float fx = flow_at_point[0] / maxrad;  //归一化光流分量
			float fy = flow_at_point[1] / maxrad;  
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))  
			{  
				data[0] = data[1] = data[2] = 0;  //
				continue;  
			}  
			float rad = sqrt(fx * fx + fy * fy);  

			float angle = atan2(-fy, -fx) / CV_PI;  
			float fk = (angle + 1.0) / 2.0 * (colorwheel.size()-1);  
			int k0 = (int)fk;  
			int k1 = (k0 + 1) % colorwheel.size();  
			float f = fk - k0;  

			for (int b = 0; b < 3; b++)   
			{  
				float col0 = colorwheel[k0][b] / 255.0;  
				float col1 = colorwheel[k1][b] / 255.0;  
				float col = (1 - f) * col0 + f * col1;  
				if (rad <= 1)  
					col = 1 - rad * (1 - col); // 增大饱和半径  
				else  
					col *= .75; // 超出范围  
				data[2 - b] = (int)(255.0 * col);  
			}  
		}  
	}  
}  

bool cmp(vector<Point> &v1,vector<Point> &v2)//将检测到的轮廓按面积大小进行排序
{
	return contourArea(v1)>contourArea(v2);
}

const float scale=0.25;//缩放系数
const int invscale=(int)1/scale;//导数是在追踪到目标后将举行画到原始图像帧中
int main(int argc, char** argv)
{
	/*从这里开始是立体标定*/
    Size boardSize;//棋盘大小
    string imagelistfn;//XML文件（将两个摄像机采集的棋盘图像的名字保存到XML文件中，方便后面读取图像）
    bool showRectified = true;//是否显示校正后的图像对

    if( imagelistfn == "" ) //读取图像名字符串
    {
        imagelistfn = "C:/Users/lenovo/Desktop/【重要】视觉程序/追踪程序/【完整】双目追踪/棋盘图/stereo_calib.xml";
        boardSize = Size(9, 6);//棋盘格规格
    }
    else if( boardSize.width <= 0 || boardSize.height <= 0 )
    {
        cout << "如果读取的XML文件中包含带有棋盘的图像，在这里也应该制定所用棋盘的大小！" << endl;
        return 0;
    }

    vector<string> imagelist;//这个容器是将读取到的图像的名字保存起来后面用
    bool ok = readStringList(imagelistfn, imagelist);//现在这里是读取每个图像,把图像名放到imagelist 容器中
    if(!ok || imagelist.empty())
    {
        cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
        return -1;
    }

    StereoCalib(imagelist, boardSize, showRectified);//立体标定;标定后得到的参数已经保存了；在立体校正时对参数读取就可以用
	/*到这里立体标定结束*/

	/*下面开始接入视频*/
	VideoCapture cap0(0),cap1(1);
	if (!cap0.isOpened())
	{
		cout<<"左边的相机/视频未打开！\n";
	}
	if (!cap1.isOpened())
	{
		cout<<"右边的相机/视频未打开！\n";
	}

	Mat prevgray, gray, flow;//光流要用到的变量
	Mat motion2color;//光流图 
	Mat frame0,frame1;//两边的图像帧
	Mat fgmask;//两个变量分别为原图像、提取前景（二值形式：前景为1，背景为0）
	int frameNum=0;//帧数
	BackgroundSubtractorMOG2 mog;//高斯背景模型对象
	Rect rect;//轮廓外部矩形边界

	/*计算直方图用到的变量*/
	Mat hsvFrame;
	vector<Mat>vecHsvFrame;
	vector<int> channels;
	channels.push_back(0);//只绘制h通道直方图
	vector<int> hueNum;
	hueNum.push_back(30);//bin=30,即30个柱形图
	vector<float>hueRanges;//范围
	hueRanges.push_back(0);
	hueRanges.push_back(180);
	MatND hist;

	while(1)
	{
		Begin://当没有运动目标时，回到这里重新采集图像
		cap0.read(frame0);
		if (frame0.empty())
		{
			cout<<"左边的图像读取失败！\n";
		}
		cap1.read(frame1);
		if(frame1.empty())
		{
			cout<<"右边的视频读取失败！\n";
		}
		imshow("原始frame0",frame0);
		imshow("原始frame1",frame1);

		vector<Mat>rectMat=StereoRectified(frame0,frame1);//立体校正后返回裁剪好的图像ROI
		imshow("校正后裁剪左边的ROI",rectMat[0]);//裁剪得到的两幅图像在这里相当于原始图像了，但最后还是要恢复到原始图像中---按比例
		imshow("校正后裁剪右边的ROI",rectMat[1]);
		resize(rectMat[1],rectMat[1],rectMat[0].size());//将左右两幅图像的尺寸resize到一样大
		cout<<rectMat[0].size()<<endl<<rectMat[1].size()<<endl;
		
		Mat halfRectMat;
		resize(rectMat[0],halfRectMat,Size(),scale,scale);
        //这里将裁剪后的图像缩小至一半，用于后边的计算光流和将图像转换至转换至HSV空间

		Mat disp=StereoMatch(rectMat[0],rectMat[1]);//立体匹配返回视差图
		imshow("视差图",disp);
		
		Mat halfframe;
		resize(disp,halfframe,Size(),scale,scale);//将原始帧尺寸转换为原来的一半
		//imshow("halfframe",halfframe);

		Mat mask(halfframe.rows,halfframe.cols,CV_8UC1);//求直方图的掩码

		/*这里使用高斯背景模型获得前景*/
		mog(halfframe,fgmask,0.01);//提取前景，0.01为学习率
		//imshow("前景",fgmask);//显示前景

		/*下面用光流法提取前景*/
		double t = (double)getTickCount();  
		cvtColor(halfRectMat, gray, CV_BGR2GRAY); //计算光流时要转换为灰度图像

		if( prevgray.data )  
		{  
			calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);  
			motionToColor(flow, motion2color);//计算光流  
		}

		std::swap(prevgray, gray);  //将gray和pregray的内容交换，相当于取两个相邻的帧，用于计算光流（计算光流时要两个相邻的帧）

		if(!motion2color.data)//没有光流图时重新采集图像
		{
			goto Begin;
		}
		imshow("flow", motion2color);

		t = (double)getTickCount() - t;  //计算光流的时间
		cout << "计算光流的时间: " << t / ((double)getTickFrequency()*1000.) <<"ms"<< endl;  

		Mat ofgray;//光流图转换为灰度图
		cvtColor(motion2color,ofgray,COLOR_BGR2GRAY);
		//imshow("ofgray",ofgray);

		threshold(ofgray,ofgray,242,255,THRESH_BINARY_INV);//阈值化处理提取前景
		//imshow("threshold",ofgray);

		Mat fgimg;//将高斯背景模型提取的前景和光流法提取的前景融合，得到最终的前景用于后续的操作
		fgimg=fgmask & ofgray;
		/*对前景进行处理*/
		medianBlur(fgimg,fgimg,5);//中值滤波
		//imshow("前景滤波",fgimg);//显示前景
		morphologyEx(fgimg,fgimg,MORPH_DILATE,element);//膨胀处理
		//imshow("前景膨胀",fgimg);//显示前景
		morphologyEx(fgimg,fgimg,MORPH_ERODE,element);//腐蚀处理
		imshow("前景腐蚀",fgimg);//显示前景

		/*查找前景的轮廓*/
		vector<vector<Point>>contours;//定义函数参数
		vector<Vec4i>hierarchy;

		findContours(fgimg,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);//查找轮廓
		if(contours.size()<1)//没有找到轮廓时重新采集图像
		{
			frameNum++;
			goto Begin;
		}
		sort(contours.begin(),contours.end(),cmp);//轮廓按面积从大到小进行排序

		cvtColor(halfRectMat,hsvFrame,COLOR_BGR2HSV);//转换至hsv空间
		vecHsvFrame.push_back(hsvFrame);

		for(size_t i=0;i<contours.size();++i)
		{
			if(contourArea(contours[i])<contourArea(contours[0])/5)//删除小轮廓
				break;
			rect=boundingRect(contours[i]);//矩形外部边界
			mask=0;
			mask(rect)=255;//rect为ROI设为白色

			calcHist(vecHsvFrame,channels,mask,hist,hueNum,hueRanges,false);//计算直方图
			double maxValue;
			minMaxLoc(hist,0,&maxValue,0,0);//获得直方图中的最大值
			hist=hist*255/maxValue;

			Mat backProject;//计算反向投影用于追踪
			calcBackProject(vecHsvFrame,channels,hist,backProject,hueRanges,1);

			Rect search=rect;//初始化跟踪的搜索框
			RotatedRect trackBox = CamShift(backProject, search,          //进行跟踪
				TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
			Rect rect2=trackBox.boundingRect();
			rect &=rect2;//去两个矩形的公共部分，使结果更精确

			Rect fullrect(rect.x*invscale,rect.y*invscale,rect.width*invscale,rect.height*invscale);//这里是将在缩小的帧中找到的目标在裁剪出来的图像中画出来
			rectangle(rectMat[0],fullrect,Scalar(0,0,255),3);
			imshow("在裁剪图像中的追踪结果",rectMat[0]);

			//int a=frame0.rows/rectMat[0].rows;//裁剪出来的图像与原始图像的比例
			//int b=frame0.cols/rectMat[0].cols;
			//Rect fullrect1(rect.x*invscale*b,rect.y*invscale*a,rect.width*invscale*b,rect.height*invscale*a);//这里是将在缩小的帧中找到的目标在从相机采集到的图像中画出来
			//rectangle(frame0,fullrect1,Scalar(0,0,255),3);
			//imshow("在原始图像中的追踪结果",frame0);
		}//end for

		char c=(char)waitKey(10);//退出
		if (c==(char)27||c=='q'||c=='Q')
		{                      
			break;
		}
	}
    return 0;
}
