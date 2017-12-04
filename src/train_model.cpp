#include <vector>
#include <iostream>


#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"


#include "modelcfg.h"
#include "ldmarkmodel.h"

using namespace std;
using namespace cv;


/******************************************************
// SplitString
// ·Ö¸î×Ö·û´®
// ÕÅ·å
// 2017.07.26
/*******************************************************/
void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
	std::string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	while (std::string::npos != pos2)
	{
		v.push_back(s.substr(pos1, pos2 - pos1));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if (pos1 != s.length())
		v.push_back(s.substr(pos1));
}

void loadImage(const string string_file,std::vector<ImageLabel>& mImageLabels){
	ifstream fin(string_file.c_str());
	string lines;
	while(getline(fin,lines)){
		vector<string>vc;
		SplitString(lines,vc," ");
		if(vc.size() != 141)
			continue;
		ImageLabel tmp;
		tmp.imagePath = vc[0];
		for(int i = 0 ; i < 4;++i){
			tmp.faceBox[i] = atoi(vc[i+1].c_str());
		}
		for(int i = 0 ; i < 68;){
			tmp.landmarkPos[i] = atoi(vc[5+2*i].c_str());
            tmp.landmarkPos[i+68] = atoi(vc[6+2*i].c_str());
            i = i+1;
		}
		mImageLabels.push_back(tmp);
	}
	fin.close();
	
}	
	
	

int main(int argc,char** argv)
{
    std::vector<ImageLabel> mImageLabels;
    loadImage(argv[1],mImageLabels);
    std::cout << "training data size: " <<  mImageLabels.size() << std::endl;


    vector<vector<int>> LandmarkIndexs;
    vector<int> LandmarkIndex1(IteraLandmarkIndex1, IteraLandmarkIndex1+LandmarkLength1);
    LandmarkIndexs.push_back(LandmarkIndex1);
    vector<int> LandmarkIndex2(IteraLandmarkIndex2, IteraLandmarkIndex2+LandmarkLength2);
    LandmarkIndexs.push_back(LandmarkIndex2);
    vector<int> LandmarkIndex3(IteraLandmarkIndex3, IteraLandmarkIndex3+LandmarkLength3);
    LandmarkIndexs.push_back(LandmarkIndex3);
    vector<int> LandmarkIndex4(IteraLandmarkIndex4, IteraLandmarkIndex4+LandmarkLength4);
    LandmarkIndexs.push_back(LandmarkIndex4);
    vector<int> LandmarkIndex5(IteraLandmarkIndex5, IteraLandmarkIndex5+LandmarkLength5);
    LandmarkIndexs.push_back(LandmarkIndex5);

    vector<int> eyes_index(eyes_indexs, eyes_indexs+4);
    Mat mean_shape(1, 2*LandmarkPointsNum, CV_32FC1, mean_norm_shape);
    //vector<HoGParam> HoGParams{{ VlHogVariant::VlHogVariantUoctti, 5, 11, 4, 1.0f },{ VlHogVariant::VlHogVariantUoctti, 5, 10, 4, 0.7f },{ VlHogVariant::VlHogVariantUoctti, 5, 8, 4, 0.4f },{ VlHogVariant::VlHogVariantUoctti, 5, 6, 4, 0.25f } };
    vector<HoGParam> HoGParams{{ VlHogVariant::VlHogVariantUoctti, 4, 11, 4, 0.9f },{ VlHogVariant::VlHogVariantUoctti, 4, 10, 4, 0.7f },{ VlHogVariant::VlHogVariantUoctti, 4, 9, 4, 0.5f },{ VlHogVariant::VlHogVariantUoctti, 4, 8, 4, 0.3f }, { VlHogVariant::VlHogVariantUoctti, 4, 6, 4, 0.2f } };
    vector<LinearRegressor> LinearRegressors(5);

    ldmarkmodel model(LandmarkIndexs, eyes_index, mean_shape, HoGParams, LinearRegressors);
    model.train(mImageLabels);
    save_ldmarkmodel(model, "PCA-SDM-model.bin");


    system("pause");
    return 0;
}
