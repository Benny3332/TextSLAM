/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <cmath>
#include "ceres/ceres.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <system.h>
#include <setting.h>
#include <tool.h>

using namespace std;
int main(int argc, char **argv)
{
    srand(0);
    TextSLAM::tool Tool;

    // 1. 输入与输出。终端：./TextSLAM ../yaml/xxx.yaml
    // 1. input & output. terminal: ./TextSLAM ../yaml/xxx.yaml
    if(argc!=2){
        throw runtime_error("wrong number of argument.");
        return 1;
    }
    string argv1 = (string)argv[1];
    TextSLAM::setting Set(argv1);

    // 2. 获取图片和文本信息
    // 2. get Img, TextInfo
    vector<string> vImg_Name, vImg_Idx;
    vector<double> vImg_Time;
    Tool.ReadImage(Set.sReadPath_ImgList, vImg_Name, vImg_Idx, vImg_Time);
    int vImg_num = vImg_Name.size();

    vector<float> vTimePerImg;
    //调整容器（在这里是vTimePerImg）大小
    vTimePerImg.resize(vImg_num);

    // 4. 系统开始
    // 4. system begin
    cv::Mat im, imUn;
    TextSLAM::system SLAM(Set.mK, &Set, vImg_num);
    for(size_t ni = 0; ni<vImg_num; ni++){

        // a) 读取文本信息
        // a) Read text info
        vector<vector<Eigen::Matrix<double,2,1>>> vTextDete;
        vector<TextSLAM::TextInfo> vTextMean;
        //从txt文件中读取每帧画面文本信息和对应png的四点坐标，ocr提取
        Tool.ReadText(Set.sReadPath + vImg_Idx[ni], vTextDete, vTextMean, Set.eExp_name, Set.Flag_noText);
        assert(vTextDete.size()==vTextMean.size());

        // b) 读取图片信息
        // b) Read image info
        im = cv::imread(Set.sReadPath + vImg_Name[ni], cv::IMREAD_COLOR);
        double tframe = vImg_Time[ni];
        if(im.empty()){
            cerr << endl << "Failed to load image at: "
                 << Set.sReadPath << vImg_Name[ni] << endl;
            exit(-1);
        }

        // c) 校正图片
        // c) undistor Img
        // Set.mKcv 相机内参矩阵；Set.mDistcv 畸变系数矩阵；Set.mKcv 校正后相机内参矩阵
        // newCameraMatrix 或者，如果你确定不需要调整视角，并且希望图像填充整个视图窗口（可能包含一些黑色边框），
        // 你可以简单地使用内参矩阵作为新相机矩阵，但请注意这可能会导致图像边缘的某些部分被裁剪掉。

        cv::undistort(im, imUn, Set.mKcv, Set.mDistcv, Set.mKcv);

        // ---- log ----
        if(ni%500==0){
            cout << "processing image: "<<vImg_Name[ni]<<endl;
            cout<<"......"<<endl;
        }
        std::chrono::steady_clock::time_point t_Begin = std::chrono::steady_clock::now();
        // ---- log ----

        // d) 将图片传递给追踪器
        // d) pass img to tracker
        SLAM.TrackMonocular(imUn,tframe, Set.mK, vTextDete, vTextMean);

        // ---- log ----
        std::chrono::steady_clock::time_point t_End = std::chrono::steady_clock::now();
        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t_End - t_Begin).count();
        vTimePerImg[ni]=ttrack;
        // ---- log ----
    }

    // 记录关键帧到"keyframe.txt"
    // "keyframe.txt"
    SLAM.RecordKeyFrame();


    return 0;
}

