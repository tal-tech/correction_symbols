//
// Created by liuxin on 20-6-24. Modified by QiaoZhi on 22-04-12
//
#pragma once

#include <string>
//#include <Python.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//include <opencv2/imgcodecs/imgcodecs_c.h> //opencv3时用这个
#include <opencv2/imgcodecs/legacy/constants_c.h>//opencv4时用这个
//#include <numpy/arrayobject.h>
#include <iostream>
#include <fstream>
#include <map>
#include <algorithm>
#include <numeric>
#include <sys/timeb.h>
#include <wchar.h>
#include <map>

//using namespace std;
namespace facethink
{
    class NumRecClassify;
}

class Classifier
{
public:
    Classifier();
    ~Classifier();

private:
    facethink::NumRecClassify *crnn_module; //识别模型

    //static std::vector<std::string> state_map;//字典

public:
    /*
     *功能:初始化识别模型
     参数：
     config_file【IN】:	配置文件config.ini
     zidian_file_path【IN】：字典txt文件
     model_file_path【IN】：模型pt文件
    return:
    true --- 初始化成功
    false---初始化失败
     */
    bool initModule(
        const std::string &config_file,
        const std::string &zidian_file_path,
        const std::string &model_file_path);

    /*
     * 功能:batch解码，得到识别结果字符串和置信度
     * 参数：
     * Cropimgs_all【IN】 --- 待解码的图,8bit灰度图
     *results【OUT】---接收解码后的字符串结果,两层vector，最外层是batch维度，里层是topk维度
     *resultweights【OUT】---接收解码后的置信度结果，两层vector，最外层是batch维度，里层是topk维度
     *resultcharweights【OUT】---接收解码每个字符对应的置信度
     *nRegColor[IN]---是否只识别红色， 0---正常识别  1---只提取红色识别
     *nFilterResult【IN】---结果过滤逻辑，0---不做任何过滤，1---过滤掉非批改字符
     *nBatchSize【IN】---识别的batch size
     *nTopK【IN】---最多获取多少个候选结果，最终返回的候选结果数量<=该值
     *返回值:
     *0---成功
     *其它---失败
     */
//    int get_batch_decode_result(std::vector<cv::Mat> &Cropimgs_all, std::vector<std::vector<std::string>> &results, std::vector<std::vector<float>> &resultweights,const int nRegColor=0,const int nFilterResult=0,const int nBatchSize = 8,const int nTopK = 1);
    int get_batch_decode_result(std::vector<cv::Mat> &Cropimgs_all,
                                std::vector<std::vector<std::string>> &results,
                                std::vector<std::vector<float>> &resultweights,
                                std::vector<std::vector<std::vector<float>>> &resultcharweights,
                                const int nRegColor=0,
                                const int nFilterResult=0,
                                const int nBatchSize = 8,
                                const int nTopK = 1);
    /*功能：对模型推理出的概率矩阵解码
     *参数:
     *Cropimgs_all【IN】---图像数据，8bit灰度图
     *arraypro【IN】---模型推理得到的概率矩阵
     *results【OUT】--接收字符串识别结果
     *regPositionsList【OUT】---接收字符的位置，帧序号
     *resultsPoints【OUT】---接收字符的位置，横坐标值
     *resultweights【OUT】---接收解码置信度
     *resultcharweights【OUT】---接收解码每个字符对应的置信度
     * nTopK【IN】---最多获取多少个候选结果，最终返回的候选结果数量<=该值
     */
//    void RunReg_batchimage_topk(std::vector<cv::Mat> &Cropimgs_all, const std::vector<std::vector<std::vector<float>>>& arraypro, std::vector<std::vector<std::string>> &results, std::vector<std::vector<std::vector<int>>> &regPositionsList, std::vector<std::vector<std::vector<cv::Point2f>>> &resultsPoints, std::vector<std::vector<float>> &resultweights,const int nTopK = 10);
    void RunReg_batchimage_topk(std::vector<cv::Mat> &Cropimgs_all,
                                const std::vector<std::vector<std::vector<float>>>& arraypro,
                                std::vector<std::vector<std::string>> &results,
                                std::vector<std::vector<std::vector<int>>> &regPositionsList,
                                std::vector<std::vector<std::vector<cv::Point2f>>> &resultsPoints,
                                std::vector<std::vector<float>> &resultweights,
                                std::vector<std::vector<std::vector<float>>> &resultcharweights,
                                const int nTopK = 10);
    /*功能：识别模型batch推理
     *参数：
     *imglist【IN] ---图像数据
     *batch【IN】---推理时设定的batch size
     *vecProbs【OUT】---接收推理返回的概率矩阵数据
     *返回值：
     *0---成功
     *其它---失败
     */
    int getbatchProMats(std::vector<cv::Mat> &imglist, const int &batch, std::vector<std::vector<std::vector<float>>>& vecProbs);
    
     
    /*功能:两个模型的识别结果融合
     *参数:
     *vec_Result1[IN] --- 第一个模型的多侯选识别结果
     *vec_weight1[IN] --- 第一个模型的多侯选识别置信度
     *vec_char_weights1[IN] --- 第一个模型的多侯选识别字符对应置信度
     *vec_Result2[IN] --- 第二个模型的多侯选识别结果
     *vec_weight2[IN] --- 第二个模型的多侯选识别结果
     *vec_char_weights2[IN] --- 第二个模型的多侯选识别字符对应置信度
     *vec_Result_ret[OUT] --- 接收融合后的识别结果
     *vec_weight_ret[OUT] --- 接收融合后的识别置信度
     *vec_char_weight_ret[OUT] --- 接收融合后的字符识别置信度
     *返回:
     * 0---成功
     * 其它---失败
    */
//    static int combiResult(const std::vector<std::string>& vec_Result1,const std::vector<float>& vec_weight1,const std::vector<std::string>& vec_Result2,const std::vector<float>& vec_weight2,std::vector<std::string>& vec_Result_ret,std::vector<float>& vec_weight_ret);
    static int combiResult(const std::vector<std::string>& vec_Result1,
                           const std::vector<float>& vec_weight1,
                           const std::vector<std::vector<float>>& vec_char_weights1,
                           const std::vector<std::string>& vec_Result2,
                           const std::vector<float>& vec_weight2,
                           const std::vector<std::vector<float>>& vec_char_weights2,
                           std::vector<std::string>& vec_Result_ret,
                           std::vector<float>& vec_weight_ret,
                           std::vector<std::vector<float>>& vec_char_weight_ret);
private:
    std::vector<std::string> state_map;  //string dict for decode
};
