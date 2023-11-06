#include <math.h>
#include <random>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

void corridorBuilder2d(float origin_x, float origin_y, float radius,
                       std::vector<Eigen::Vector2f> &data,
                       std::vector<cv::Point2f> &finalVertex,
                       std::vector<Eigen::Vector3f> &outputConstrains)
{
    outputConstrains.clear();
    finalVertex.clear();
    float interior_x = 0.0;
    float interior_y = 0.0;
    float safe_radius = radius; // 初始安全半径
    std::vector<cv::Point2f> flipData(data.size() + 1, cv::Point2f(0, 0));

    // 1.sphere flipping transformation
    for (size_t i = 0; i < data.size(); i++)
    {
        float dx = data[i](0) - origin_x;
        float dy = data[i](1) - origin_y;
        float norm2 = std::sqrt(dx * dx + dy * dy);
        // 更新安全半径
        if (norm2 < safe_radius)
            safe_radius = norm2;
        if (norm2 == 0)
            continue;
        flipData[i].x = dx + 2 * (radius - norm2) * dx / norm2;
        flipData[i].y = dy + 2 * (radius - norm2) * dy / norm2;
    }

    // 2.构建凸包
    std::vector<int> vertexIndice;
    cv::convexHull(flipData, vertexIndice, false, false);

    // 3.提取出凸包顶点对应的实际数据data中的点-->vertexData，并检查origin是否也是一个凸包顶点
    bool isOriginAVertex = false;
    int OriginIndex = -1;
    std::vector<cv::Point2f> vertexData;
    for (size_t i = 0; i < vertexIndice.size(); i++)
    {
        int v = vertexIndice[i];
        if (v == (int)(data.size()))
        {
            isOriginAVertex = true;
            OriginIndex = i;
            vertexData.push_back(cv::Point2f(origin_x, origin_y));
        }
        else
        {
            vertexData.push_back(cv::Point2f(data[v](0), data[v](1)));
        }
    }

    // 4.如果origin也是一个顶点，就找到原点周围的一个合适的内部点。
    if (isOriginAVertex)
    {
        int last_index = (OriginIndex - 1) % vertexIndice.size();
        int next_index = (OriginIndex + 1) % vertexIndice.size();
        float dx = (data[vertexIndice[last_index]](0) + origin_x + data[vertexIndice[next_index]](0)) / 3 - origin_x;
        float dy = (data[vertexIndice[last_index]](1) + origin_y + data[vertexIndice[next_index]](1)) / 3 - origin_y;
        float d = std::sqrt(dx * dx + dy * dy);
        interior_x = 0.99 * safe_radius * dx / d + origin_x;
        interior_y = 0.99 * safe_radius * dy / d + origin_y;
    }
    else
    {
        interior_x = origin_x;
        interior_y = origin_y;
    }

    // 5. 根据球形翻转构建的凸包结果，在实际点上构造一个初始的内部多面体
    std::vector<int> vIndex2;
    cv::convexHull(vertexData, vIndex2, false, false); // counterclockwise right-hand

    // 6.对星凸多面体的每个顶点构建等式约束，确定相关法向量方向上距离原点最近的星凸顶点
    std::vector<Eigen::Vector3f> constraints; // (a,b,c) a x + b y <= c
    for (size_t j = 0; j < vIndex2.size(); j++)
    {
        int jplus1 = (j + 1) % vIndex2.size();
        cv::Point2f rayV = vertexData[vIndex2[jplus1]] - vertexData[vIndex2[j]];
        Eigen::Vector2f normalJ(rayV.y, -rayV.x); // point to outside
        normalJ.normalize();
        int indexJ = vIndex2[j];
        while (indexJ != vIndex2[jplus1])
        {
            float c = (vertexData[indexJ].x - interior_x) * normalJ(0) + (vertexData[indexJ].y - interior_y) * normalJ(1);
            constraints.push_back(Eigen::Vector3f(normalJ(0), normalJ(1), c));
            indexJ = (indexJ + 1) % vertexData.size();
        }
    }

    std::vector<cv::Point2f> dualPoints(constraints.size(), cv::Point2f(0, 0));
    for (size_t i = 0; i < constraints.size(); i++)
    {
        dualPoints[i].x = constraints[i](0) / constraints[i](2);
        dualPoints[i].y = constraints[i](1) / constraints[i](2);
    }

    std::vector<cv::Point2f> dualVertex;
    cv::convexHull(dualPoints, dualVertex, true, false);
    std::cout << dualPoints.size() << ", " << dualVertex.size() << std::endl;

    for (size_t i = 0; i < dualVertex.size(); i++)
    {
        int iplus1 = (i + 1) % dualVertex.size();
        cv::Point2f rayi = dualVertex[iplus1] - dualVertex[i];
        float c = rayi.y * dualVertex[i].x - rayi.x * dualVertex[i].y;
        finalVertex.push_back(cv::Point2f(interior_x + rayi.y / c, interior_y - rayi.x / c));
    }

    // finalVertex: 最后的严格凸的多面体对应的顶点
    for (size_t i = 0; i < finalVertex.size(); i++)
    {
        int iplus1 = (i + 1) % finalVertex.size();
        cv::Point2f rayi = finalVertex[iplus1] - finalVertex[i];
        float c = rayi.y * finalVertex[i].x - rayi.x * finalVertex[i].y;
        outputConstrains.push_back(Eigen::Vector3f(rayi.y, -rayi.x, c));
    }
}