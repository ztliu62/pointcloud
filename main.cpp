#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Geometry>

#include <boost/format.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>

#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/io/vtk_io.h>
using namespace std;

int main(int argc, char** argv) {

    vector<cv::Mat> colorImgs, depthImgs;
    vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses;

    ifstream fin("../pose.txt");
    if (!fin){
        cerr << "Please get pose.txt file" << endl;
        return 1;
        }

    for (int i = 0; i < 5 ; i++){
        boost::format fmt("../%s/%d.%s");
        cv::Mat colorImg = cv::imread((fmt%"color"%(i+1)%"png").str());
        cv::Mat depthImg = cv::imread((fmt%"depth"%(i+1)%"pgm").str(), -1);
        //cout << colorImg.rows << " " << colorImg.cols << endl;

        colorImgs.push_back(colorImg);
        depthImgs.push_back(depthImg);

        double data[7] = {0};
        for (auto& d : data){
            fin >> d;
        }

        Eigen::Quaterniond q (data[6], data[3], data[4], data[5]);
        Eigen::Isometry3d T(q);
        T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(T);
    }

    double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
    double depthScale = 1000.0;

    cout << "Converting image to Point Cloud ... " << endl;
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

    PointCloud::Ptr pointcloud(new PointCloud);
    for (int i = 0; i < 5; i++){
        cout << "Converting the Image " << i+1 << endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Eigen::Isometry3d T = poses[i];
        for (int v = 0; v < color.rows; v++){
            for (int u = 0; u < color.cols; u++){
                auto d = depth.ptr<unsigned short>(v)[u];
                if (d == 0){
                    continue;
                }

                Eigen::Vector3d point;
                point[2] = double(d)/depthScale;
                point[0] = (u - cx)*point[2]/fx;
                point[1] = (v - cy)*point[2]/fy;
                Eigen::Vector3d pointWorld = T*point;

                PointT p;
                p.x = pointWorld[0];
                p.y = pointWorld[1];
                p.z = pointWorld[2];
                p.b = color.data[v*color.step + u*color.channels()];
                p.g = color.data[v*color.step + u*color.channels() + 1];
                p.r = color.data[v*color.step + u*color.channels() + 2];
                pointcloud->points.push_back(p);

            }
        }

    }
    pointcloud->is_dense = false;
    cout << "Totally " << pointcloud->size() << " Points in Point Clouds" << endl;
    pcl::io::savePCDFileBinary("map.pcd", *pointcloud);

    pcl::VoxelGrid<PointT> downsampled;
    PointCloud::Ptr filtered_cloud(new PointCloud);
    downsampled.setInputCloud(pointcloud);
    downsampled.setLeafSize(0.01f, 0.01f, 0.01f);
    downsampled.filter(*filtered_cloud);
    cout << "Totally " << filtered_cloud->size() << " Points in Filtered Point Clouds" << endl;
    pcl::io::savePCDFileBinary("filtered_map.pcd", *filtered_cloud);


    //pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    //pcl::copyPointCloud(*filtered_cloud, *xyz_cloud);

    pcl::StatisticalOutlierRemoval<PointT> sor;
    PointCloud::Ptr cloud_sor(new PointCloud);
    sor.setInputCloud(filtered_cloud);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*cloud_sor);
    cout << "Implementing Statistical Outlier Removal" << endl;
    pcl::io::savePCDFileBinary("sor_map.pcd", *cloud_sor);

    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    PointCloud::Ptr mls_points(new PointCloud);
    pcl::MovingLeastSquares<PointT, PointT> mls;
    mls.setComputeNormals(false);
    mls.setInputCloud(cloud_sor);
    mls.setPolynomialOrder(2);
    mls.setSearchMethod(tree);
    mls.setSearchRadius(0.08);
    mls.process(*mls_points);
    cout << "Implementing Moving Least Square" << endl;
    pcl::io::savePCDFileBinary("mls_map.pcd", *mls_points);

    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    pcl::search::KdTree<PointT>::Ptr ntree(new pcl::search::KdTree<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ntree->setInputCloud(mls_points);
    ne.setInputCloud(mls_points);
    ne.setSearchMethod(ntree);
    ne.setKSearch(20);
    ne.compute(*normals);
    cout << "Calculating Pointcloud Normal" << endl;

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::concatenateFields(*mls_points, *normals, *cloud_normals);
    pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr ntree2 (new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
    ntree2->setInputCloud(cloud_normals);
    pcl::GreedyProjectionTriangulation<pcl::PointXYZRGBNormal> gp3;
    pcl::PolygonMesh triangles;
    gp3.setSearchRadius(0.05);
    gp3.setMu(2.5);
    gp3.setMaximumNearestNeighbors(100);
    gp3.setMaximumSurfaceAngle(M_PI/4);
    gp3.setMaximumAngle(2*M_PI/3);
    gp3.setNormalConsistency(false);

    gp3.setInputCloud(cloud_normals);
    gp3.setSearchMethod(ntree2);
    gp3.reconstruct(triangles);

    pcl::io::saveVTKFile("mesh.vtk", triangles);

    return 0;
}