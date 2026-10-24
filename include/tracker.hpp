// tracker.hpp - Kelas-kelas untuk tracking objek berwarna + estimasi posisi 3D
//
//  CameraParams  : parameter kamera (fx, fy, cx, cy) yang dibaca dari file YAML
//  Detection     : hasil deteksi satu objek pada satu frame
//  SharedParams  : parameter yang dipakai bersama oleh semua objek (diatur via trackbar)
//  ObjectTracker : satu objek fisik -> threshold HLS, morfologi, kontur, bounding box, X Y Z

#pragma once

#include <opencv2/opencv.hpp>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// Parameter kamera hasil kalibrasi (dibaca dinamis dengan cv::FileStorage)
// ---------------------------------------------------------------------------
class CameraParams
{
public:
    double fx = 0.0, fy = 0.0; // focal length (piksel)
    double cx = 0.0, cy = 0.0; // principal point (piksel)
    int width = 0, height = 0; // resolusi saat kalibrasi

    bool load(const std::string &path)
    {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            std::cerr << "Gagal membuka file kalibrasi: " << path << std::endl;
            return false;
        }

        cv::Mat K;
        fs["camera_matrix"] >> K;
        fs["image_width"] >> width;
        fs["image_height"] >> height;
        fs.release();

        if (K.empty() || K.rows != 3 || K.cols != 3)
        {
            std::cerr << "camera_matrix tidak valid di " << path << std::endl;
            return false;
        }

        K.convertTo(K, CV_64F);
        fx = K.at<double>(0, 0);
        fy = K.at<double>(1, 1);
        cx = K.at<double>(0, 2);
        cy = K.at<double>(1, 2);

        return fx > 0.0 && fy > 0.0;
    }
};

// ---------------------------------------------------------------------------
// Hasil deteksi satu objek
// ---------------------------------------------------------------------------
struct Detection
{
    bool found = false;
    cv::Rect box;          // bounding box
    cv::Point2d centroid;  // (x_pixel, y_pixel)
    double X = 0.0;        // satuan sama dengan W_real (cm)
    double Y = 0.0;
    double Z = 0.0;
};

// ---------------------------------------------------------------------------
// Parameter bersama (diubah lewat trackbar)
// ---------------------------------------------------------------------------
struct SharedParams
{
    int minArea = 1000;   // luas kontur minimum agar dianggap objek
    int erodeIter = 2;    // iterasi erode
    int dilateIter = 6;   // iterasi dilate
    int warnDistCm = 30;  // jarak (cm) di bawah ini -> proximity warning
};

// ---------------------------------------------------------------------------
// Satu objek fisik yang dilacak
// ---------------------------------------------------------------------------
class ObjectTracker
{
public:
    std::string name;
    double wRealCm;        // lebar fisik sebenarnya (cm)
    cv::Scalar drawColor;  // warna bounding box (BGR)

    // Batas threshold HLS. Dibuat public karena dihubungkan langsung ke trackbar.
    // Jika hLow > hHigh, rentang Hue dianggap melewati batas 179 -> 0 (cocok untuk merah).
    int hLow, hHigh;
    int lLow, lHigh;
    int sLow, sHigh;

    ObjectTracker(const std::string &name, double wRealCm, const cv::Scalar &drawColor,
                  int hLow, int hHigh, int lLow, int lHigh, int sLow, int sHigh)
        : name(name), wRealCm(wRealCm), drawColor(drawColor),
          hLow(hLow), hHigh(hHigh), lLow(lLow), lHigh(lHigh), sLow(sLow), sHigh(sHigh)
    {
    }

    // Buat jendela trackbar untuk menyetel threshold HLS objek ini.
    // PENTING: objek tidak boleh dipindah/di-copy setelah fungsi ini dipanggil,
    // karena trackbar menyimpan alamat variabel anggota di atas.
    void createTrackbars()
    {
        const std::string win = "Threshold - " + name;
        cv::namedWindow(win, cv::WINDOW_AUTOSIZE);
        cv::createTrackbar("H low", win, &hLow, 179);
        cv::createTrackbar("H high", win, &hHigh, 179);
        cv::createTrackbar("L low", win, &lLow, 255);
        cv::createTrackbar("L high", win, &lHigh, 255);
        cv::createTrackbar("S low", win, &sLow, 255);
        cv::createTrackbar("S high", win, &sHigh, 255);
    }

    // Proses satu frame (sudah dalam format HLS) dan kembalikan hasil deteksi.
    Detection detect(const cv::Mat &hlsImage, const SharedParams &params, const CameraParams &cam)
    {
        buildMask(hlsImage);
        cleanMask(params);

        Detection det;

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Ambil kontur terbesar
        int best = -1;
        double bestArea = 0.0;
        for (size_t i = 0; i < contours.size(); i++)
        {
            double area = cv::contourArea(contours[i]);
            if (area > bestArea)
            {
                bestArea = area;
                best = (int)i;
            }
        }

        // Filter luas minimum: noise kecil di latar belakang diabaikan
        if (best < 0 || bestArea < params.minArea)
            return det;

        det.found = true;
        det.box = cv::boundingRect(contours[best]);

        // Centroid dari momen kontur (fallback: titik tengah bounding box)
        cv::Moments m = cv::moments(contours[best]);
        if (m.m00 > 0.0)
            det.centroid = cv::Point2d(m.m10 / m.m00, m.m01 / m.m00);
        else
            det.centroid = cv::Point2d(det.box.x + det.box.width / 2.0,
                                       det.box.y + det.box.height / 2.0);

        // Estimasi koordinat 3D (Pinhole Camera Model)
        //   Z = fx * W_real / w_pixel
        //   X = (x_pixel - cx) * Z / fx
        //   Y = (y_pixel - cy) * Z / fy
        double wPixel = (double)det.box.width;
        det.Z = (cam.fx * wRealCm) / wPixel;
        det.X = ((det.centroid.x - cam.cx) * det.Z) / cam.fx;
        det.Y = ((det.centroid.y - cam.cy) * det.Z) / cam.fy;

        return det;
    }

    const cv::Mat &getMask() const { return mask; }

private:
    cv::Mat mask;

    // Color threshold (cv::inRange) pada gambar HLS. Urutan Scalar = (H, L, S).
    void buildMask(const cv::Mat &hlsImage)
    {
        if (hLow <= hHigh)
        {
            cv::inRange(hlsImage, cv::Scalar(hLow, lLow, sLow), cv::Scalar(hHigh, lHigh, sHigh), mask);
        }
        else
        {
            // Hue melewati batas: [hLow..179] gabung [0..hHigh]
            cv::Mat m1, m2;
            cv::inRange(hlsImage, cv::Scalar(hLow, lLow, sLow), cv::Scalar(179, lHigh, sHigh), m1);
            cv::inRange(hlsImage, cv::Scalar(0, lLow, sLow), cv::Scalar(hHigh, lHigh, sHigh), m2);
            cv::bitwise_or(m1, m2, mask);
        }
    }

    // Morfologi: erode (hapus noise kecil) -> dilate (isi lubang) -> erode (kembalikan ukuran)
    void cleanMask(const SharedParams &p)
    {
        int e = std::max(0, p.erodeIter);
        int d = std::max(0, p.dilateIter);

        cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), e);
        cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), d);
        cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), std::max(0, d - e));
    }
};