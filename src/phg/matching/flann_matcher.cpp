#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
    index_params = flannKdTreeIndexParams(4);
    search_params = flannKsTreeSearchParams(32);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    cv::Mat indices;
    cv::Mat dists;

    flann_index->knnSearch(query_desc, indices, dists, k, *search_params);

    matches.resize(indices.rows, std::vector<cv::DMatch>(k));
    for (int i = 0; i < query_desc.rows; i++) {
        for (int j = 0; j < k; j++) {
            matches[i][j].queryIdx = i;
            matches[i][j].trainIdx = indices.at<int>(i, j);
            matches[i][j].distance = sqrt(dists.at<float>(i, j));
        }
    }
}
