#include <iomanip>

#include <vector>
#include <stdio.h>
#include<chrono>

#include <ctime> 
#include<cmath>
#include <opencv2/core/mat.hpp>
#include <fstream>
#include <iostream>
#include "reg_pigai.h"
//#include <torch/script.h>
int main(int argc, char* argv[])
{
   
    if (argc< 9) // 4
    {
        std::cout << "input param not enough!eg.\n 1. xx image_folder reg_batch run_time config pt_1 pt_2 voc_1 voc_2" << std::endl;
	return 0;
    }
    std::string filepath = std::string(argv[1]);
    int nRegBatchs = atoi(argv[2]);
    if (nRegBatchs <1)
    {
        std::cout<<"please specify the correct reg batch"<<std::endl;
        return 0;
    }
    int run_times = atoi(argv[3]);
    std::string config_path = std::string(argv[4]); // config文件的路径
    std::string pt_common_ocr = std::string(argv[5]); // 通用OCR模型的pt文件
    std::string pt_pigai_ocr = std::string(argv[6]); // 批改识别模型的pt文件
    std::string voc_common_ocr = std::string(argv[7]); // 通用识别模型的字典
    std::string voc_pigai_ocr = std::string(argv[8]); // 批改识别模型的字典

    //通用识别模型
    const std::string model_file_path1 = pt_common_ocr; // "./sdk/model/crnn_Rec_swa_462w_blur_8_16_20210524.pt";
    const std::string config_file1 = config_path; // "./sdk/model/config.ini";
    const std::string zidian_file_path1 = voc_common_ocr; // "./sdk/model/zidian_new_5883.txt";


    //批改识别模型
    const std::string model_file_path2 = pt_pigai_ocr; // "./sdk/model/crnn_best_acc94.pt";
    const std::string config_file2 = config_path; // "./sdk/model/config.ini";
    const std::string zidian_file_path2 = voc_pigai_ocr; // "./sdk/model/dict_henji.txt";

	// 初始化两个识别模型
    Classifier *reger1 = new Classifier();
    bool ifreg1 = reger1->initModule(config_file1, zidian_file_path1, model_file_path1);
    if(!ifreg1)
    { 
        std::cout<<"init tongyong OCR module failed!"<<std::endl;
	return 0;
    }
    Classifier *reger2 = new Classifier();
    bool ifreg2 = reger2->initModule(config_file2, zidian_file_path2, model_file_path2);
    if(!ifreg2)
    {
        std::cout<<"init pigai OCR module failed!"<<std::endl;
	return 0;
    }


    int t =0;
    bool bFirstTime = true;    
    std::vector<cv::String> files;
    cv::glob(filepath,files,false);
    while(run_times < 0 || t < run_times)
    {
        std::vector<cv::Mat> imglist;
        std::vector<float> vecAllTime;
        std::vector<std::string> vecImgNames;
        int batch = nRegBatchs;
        std::ofstream out("./result.txt");
        for(int i = 0;i<files.size();i++){
            cv::Mat image = cv::imread(files[i]);
	        if(image.cols <= 0 || image.rows <= 0 || image.rows >= 4096 || image.cols >= 4096 || image.rows > 20*image.cols){
                continue;
            }
	        size_t pos_seg = files[i].rfind("/");
	        std::string img_name = files[i];
	        if(pos_seg != std::string::npos){
                img_name = files[i].substr(pos_seg+1);
	        }
            vecImgNames.emplace_back(img_name);
            // cv::Mat gray_crop;
            // cvtColor(image, gray_crop, cv::COLOR_BGR2GRAY);
            imglist.push_back(image.clone());
            if(imglist.size()%batch==0 || (i+1 >= files.size() && imglist.size() > 0))
            {
                std::vector<std::vector<std::string>> results;
                std::vector<std::vector<float>> resultweights;
                std::vector<std::vector<std::vector<float>>> resultcharweights;
                std::vector<std::vector<std::string>> results1;
                std::vector<std::vector<float>> resultweights1;
                std::vector<std::vector<std::vector<float>>> resultcharweights1;
                std::vector<std::vector<std::string>> results2;
                std::vector<std::vector<float>> resultweights2;
                std::vector<std::vector<std::vector<float>>> resultcharweights2;

                auto t1 = std::chrono::system_clock::now();
                auto tstart = t1;
                // 调用通用识别
                // Params: 图像, 识别结果, 识别结果置信度, 识别结果字符对应置信度, 是否提取红色前处理(0-否), 是否清理识别结果(0-否), Batch Size, TopK
                reger1->get_batch_decode_result(imglist, results1, resultweights1, resultcharweights1, 0, 0, batch, 1);
                //调用批改符号识别
                reger2->get_batch_decode_result(imglist, results2, resultweights2, resultcharweights2, 0, 0, batch, 1);
                for(int nIdx =0; nIdx < results1.size() || nIdx < results2.size(); nIdx++){
                    std::vector<std::string> vec_result1;
                    std::vector<std::string> vec_result2;
                    std::vector<std::string> vec_result;
                    std::vector<float> vec_weight1;
                    std::vector<float> vec_weight2;
                    std::vector<float> vec_weight;
                    std::vector<std::vector<float>> vec_char_weight1;
                    std::vector<std::vector<float>> vec_char_weight2;
                    std::vector<std::vector<float>> vec_char_weight;
                    if(nIdx < results1.size()){
                        vec_result1=results1[nIdx];
                        vec_weight1=resultweights1[nIdx];
                        vec_char_weight1 = resultcharweights1[nIdx];
                    }
                    if(nIdx < results2.size())
                    {
                        vec_result2=results2[nIdx];
                        vec_weight2=resultweights2[nIdx];
                        vec_char_weight2 = resultcharweights2[nIdx];
                    }
                    //识别结果融合
                    Classifier::combiResult(vec_result1, vec_weight1, vec_char_weight1, vec_result2, vec_weight2, vec_char_weight2, vec_result, vec_weight, vec_char_weight);
                    results.emplace_back(vec_result);
                    resultweights.emplace_back(vec_weight);
                    resultcharweights.emplace_back(vec_char_weight);
                }
                auto t2 = std::chrono::system_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
                auto time_consuming = duration * 1e-3;
                if(imglist.size()%batch==0 && !bFirstTime){
                    vecAllTime.emplace_back(time_consuming);
                }
                bFirstTime = false;
                for(int nBIndex = 0; nBIndex < results.size();nBIndex++){
                    if(results[nBIndex].size() > 0){
                        std::cout<<vecImgNames[nBIndex]<<"\t"<<results[nBIndex][0]<<"\t"<<resultweights[nBIndex][0]<<"\t";
                        out<<vecImgNames[nBIndex]<<"\t"<<results[nBIndex][0]<<"\t"<<resultweights[nBIndex][0]<<"\t";
                        for (size_t char_idx=0; char_idx<resultcharweights[nBIndex][0].size(); char_idx++){
                            std::cout<<resultcharweights[nBIndex][0][char_idx]<<" ";
                            out<<resultcharweights[nBIndex][0][char_idx]<<" ";
                        }
                        std::cout<<std::endl;
                        out<<std::endl;
                    }
                    else
                    {
                        std::cout<<vecImgNames[nBIndex]<<std::endl;
                        out<<vecImgNames[nBIndex]<<std::endl;
                    }
                }
                imglist.clear();
                vecImgNames.clear();
            }
        }
	    if(vecAllTime.size() > 1){
            std::sort(vecAllTime.begin(),vecAllTime.end());
        }
        float tm_sum = 0;
        for(int nTIndex =0; nTIndex < vecAllTime.size();nTIndex++){
            tm_sum += vecAllTime[nTIndex];
        }
        if(vecAllTime.size()> 0){
            std::cout<<"mean time:"<<tm_sum/vecAllTime.size()<<std::endl;
            std::cout<<"P90 time:"<<vecAllTime[(int)(0.9*vecAllTime.size())]<<std::endl;
            std::cout<<"P95 time:"<<vecAllTime[(int)(0.95*vecAllTime.size())]<<std::endl;
            std::cout<<"P99 time:"<<vecAllTime[(int)(0.99*vecAllTime.size())]<<std::endl;
            std::cout<<"max time:"<<vecAllTime[vecAllTime.size()-1]<<std::endl;
        }
        out.close();
        if(run_times > 0){
            t++;
        }
    }
    std::cout<<"over"<<std::endl;
    delete reger1;
    delete reger2;
    return 0;
}

