//============================================================================================-----
//== NaturalPoint Tracking Tools API Sample: Accessing Camera, Marker, and RigidBody Information
//==
//== This command-line application loads a Tracking Tools Project, lists cameras, and 3d marker
//== count.
//============================================================================================-----

#include <windows.h>
#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
#include <fstream>

#include <NPTrackingTools.h>

#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <opencv2/imgproc/imgproc.hpp>

#include "time.h"


//using namespace std;

#define DTTMFMT "%Y-%m-%d %H:%M:%S "
#define DTTMSZ 21
static char* getDtTm(char* buff) {
    time_t t = time(0);
    strftime(buff, DTTMSZ, DTTMFMT, localtime(&t));
    return buff;
}

using namespace cv;

// Local function prototypes
void CheckResult(NPRESULT result);

// Local constants
const float kRadToDeg = 0.0174532925f;

// Local class definitions
class Point4
{
public:
    Point4(float x, float y, float z, float w);

    float           operator[](int idx) const { return mData[idx]; }
    const float* Data() const { return mData; }

private:
    float           mData[4];
};

class TransformMatrix
{
public:
    TransformMatrix();

    TransformMatrix(float m11, float m12, float m13, float m14,
        float m21, float m22, float m23, float m24,
        float m31, float m32, float m33, float m34,
        float m41, float m42, float m43, float m44);

    void            SetTranslation(float x, float y, float z);
    void            Invert();

    TransformMatrix operator*(const TransformMatrix& rhs);
    Point4          operator*(const Point4& v);

    static TransformMatrix RotateX(float rads);
    static TransformMatrix RotateY(float rads);
    static TransformMatrix RotateZ(float rads);

private:
    float           mData[4][4];
};

// Main application
int main(int argc, char** argv[])
{
    printf("== NaturalPoint Tracking Tools API Marker Sample =======---\n");
    printf("== (C) NaturalPoint, Inc.\n\n");

    printf("Initializing NaturalPoint Devices\n");
    TT_Initialize();

    // Do an update to pick up any recently-arrived cameras.
    TT_Update();

    // Load a project file from the executable directory.
    printf( "Loading Project: project.ttp\n\n" );
    CheckResult( TT_LoadProject("C:\\Users\\fares\\source\\repos\\Motion vectors test\\Motion vectors test\\Debug\\project.ttp") );

    // List all detected cameras.
    printf("Cameras:\n");
    for (int i = 0; i < TT_CameraCount(); i++)
    {
        printf("\t%s\n", TT_CameraName(i));
    }
    printf("\n");

    // List all defined rigid bodies.
    printf("Rigid Bodies:\n");
    for (int i = 0; i < TT_RigidBodyCount(); i++)
    {
        printf("\t%s\n", TT_RigidBodyName(i));
    }
    printf("\n");

    // URL where the Tello sends its video stream to.
    const char* const TELLO_STREAM_URL{ "udp://0.0.0.0:11111" };

    // Tello tello{};
    // if (!tello.Bind())
    // {
    //     return 0;
    // }

    // tello.SendCommand("streamon");
    // while (!(tello.ReceiveResponse()))
    //     ;

    // VideoCapture capture{TELLO_STREAM_URL, CAP_FFMPEG};

    // // Take-off first
    // tello.SendCommand("takeoff");
    // while (!(tello.ReceiveResponse()))
    //     ;

    // bool busy{false};


    int frameCounter = 0;

    VideoCapture cap(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 960);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
    // cap.set(CV_CAP_PROP_FOURCC, 0x32595559);
    // cap.set(CV_CAP_PROP_FPS, 25);
    if (!cap.isOpened()) {
        std::cerr << "BAD" << std::endl;
    }
    else {
        std::cout << "GOOD" << std::endl;
    }

    // waitKey(5000);


    char buff[DTTMSZ];


    std::ofstream timestamp_file("timestamp_log.txt", std::ios::app);

    printf("System Is Ready, Press Any Key To Start Recording, Repress To Stop \n");

    for (int i = 10; i > 0; i--) {
        printf("starting in %d\n", i);
        Sleep(1000);
    }
    

    // Poll API data until the user hits a keyboard key.
    while (!_kbhit())
    {
        //waitKey(100);
        if (TT_Update() == NPRESULT_SUCCESS)
        {
            printf("file was read successfully. starting to read frame...\n");
            frameCounter++;
            printf("in frame %d\n", frameCounter);
            float   yaw, pitch, roll;
            float   x, y, z;
            float   qx, qy, qz, qw;
            bool    tracked;

            Mat frame;
            cap >> frame;
            
            printf("frame was read.\n");
            
            // Show what the Webcam sees
            resize(frame, frame, Size(), 0.75, 0.75);
            //imshow("Stream", frame);
            
            imwrite("C:\\Users\\fares\\Desktop\\markers.cpp files\\frame" + std::to_string(frameCounter) + ".jpg", frame);

            printf("Frame #%d: (Markers: %d)\n", frameCounter, TT_FrameMarkerCount());

            timestamp_file << getDtTm(buff) << "\n";

            for (int i = 0; i < TT_RigidBodyCount(); i++)
            {
                //std::ofstream pos_rigid_file("C:\\Users\\fares\\Desktop\\markers.cpp files\\pos_rigid_drone" + std::to_string(i) + ".csv", std::ios::app);
                std::ofstream rot_rigid_file("C:\\Users\\fares\\Desktop\\markers.cpp files\\rot_rigid_drone" + std::to_string(i) + ".csv", std::ios::app);

                TT_RigidBodyLocation(i, &x, &y, &z, &qx, &qy, &qz, &qw, &yaw, &pitch, &roll);

                if (TT_IsRigidBodyTracked(i))
                {
                    printf("%s: Pos (%.3f, %.3f, %.3f) Orient (%.1f, %.1f, %.1f)\n", TT_RigidBodyName(i),
                        x, y, z, yaw, pitch, roll);

                  //  pos_rigid_file << x << "," << y << "," << z << "\n";
                    rot_rigid_file << yaw << "," << pitch << "," << roll << "\n";

                    TransformMatrix xRot(TransformMatrix::RotateX(pitch * kRadToDeg));
                    TransformMatrix yRot(TransformMatrix::RotateY(yaw * kRadToDeg));
                    TransformMatrix zRot(TransformMatrix::RotateZ(roll * kRadToDeg));

                    // Compose the local-to-world rotation matrix in XYZ (pitch, yaw, roll) order.

                    TransformMatrix worldTransform = xRot * yRot * zRot;

                    // Inject world-space coordinates of the origin.

                    worldTransform.SetTranslation(x, y, z);

                    // Invert the transform matrix to convert from a local-to-world to a world-to-local.

                    worldTransform.Invert();

                    printf(">> Compare local rigid body coordinates with world-to-local converted markers\n");

                    float   mx, my, mz;
                    float   tx, ty, tz;

                    int     markerCount = TT_RigidBodyMarkerCount(i);
                    for (int j = 0; j < markerCount; ++j)
                    {

                        // Get the world-space coordinates of each rigid body marker.
                        TT_RigidBodyPointCloudMarker(i, j, tracked, mx, my, mz);

                        std::ofstream marker_pos_log("marker" + std::to_string(i) + std::to_string(j) + "_pos.csv", std::ios::app);
                        marker_pos_log << mx << "," << my << "," << mz << "\n";
                        marker_pos_log.close();


                        // Get the rigid body's local coordinate for each marker.
                        TT_RigidBodyMarker(0, j, &tx, &ty, &tz);

                        // Transform the rigid body point from world coordinates to local rigid body coordinates.
                        // Any world-space point can be substituted here to transform it into the local space of
                        // the rigid body.

                        Point4  worldPnt(mx, my, mz, 1.0f);
                        Point4  localPnt = worldTransform * worldPnt;

                        printf("  >> %d: Local: (%.3f, %.3f, %.3f) World-To-Local: (%.3f, %.3f, %.3f)\n", j + 1,
                            tx, ty, tz, localPnt[0], localPnt[1], localPnt[2]);

                    }

                    //== Invert the transform matrix so we can perform local-to-world.

                    worldTransform.Invert();

                    printf(">> Compare world markers with local-to-world converted rigid body markers\n");

                    for (int j = 0; j < markerCount; ++j)
                    {
                        // Get the world-space coordinates of each rigid body marker.
                        TT_RigidBodyPointCloudMarker(i, j, tracked, mx, my, mz);

                        // Get the rigid body's local coordinate for each marker.
                        TT_RigidBodyMarker(0, j, &tx, &ty, &tz);

                        // Transform the rigid body's local point to world coordinates.
                        // Any local-space point can be substituted here to transform it into world coordinates.

                        Point4  localPnt(tx, ty, tz, 1.0f);
                        Point4  worldPnt = worldTransform * localPnt;

                        printf("  >> %d: World (%.3f, %.3f, %.3f) Local-To-World: (%.3f, %.3f, %.3f)\n", j + 1,
                            mx, my, mz, worldPnt[0], worldPnt[1], worldPnt[2]);
                    }

                    printf("\n");

                }
                else
                {
                    printf("\t%s: Not Tracked\n", TT_RigidBodyName(i));
                }
            }
        }
        Sleep(40);
    }

    printf("Shutting down NaturalPoint Tracking Tools\n");
    CheckResult(TT_Shutdown());

    printf("Complete\n");
    while (!_kbhit())
    {
        Sleep(20);
    }

    return 0;
}



void CheckResult(NPRESULT result)   //== CheckResult function will display errors and ---
                                      //== exit application after a key is pressed =====---
{
    if (result != NPRESULT_SUCCESS)
    {
        // Treat all errors as failure conditions.
        printf("Error: %s\n\n(Press any key to continue)\n", TT_GetResultString(result));
        std::cout << result << std::endl;

        Sleep(20);
        exit(1);
    }
}

//
// Point4
//

Point4::Point4(float x, float y, float z, float w)
{
    mData[0] = x;
    mData[1] = y;
    mData[2] = z;
    mData[3] = w;
}

//
// TransformMatrix
//

TransformMatrix::TransformMatrix()
{
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            if (i == j)
            {
                mData[i][j] = 1.0f;
            }
            else
            {
                mData[i][j] = 0.0f;
            }
        }
    }
}

TransformMatrix::TransformMatrix(float m11, float m12, float m13, float m14,
    float m21, float m22, float m23, float m24,
    float m31, float m32, float m33, float m34,
    float m41, float m42, float m43, float m44)
{
    mData[0][0] = m11;
    mData[0][1] = m12;
    mData[0][2] = m13;
    mData[0][3] = m14;
    mData[1][0] = m21;
    mData[1][1] = m22;
    mData[1][2] = m23;
    mData[1][3] = m24;
    mData[2][0] = m31;
    mData[2][1] = m32;
    mData[2][2] = m33;
    mData[2][3] = m34;
    mData[3][0] = m41;
    mData[3][1] = m42;
    mData[3][2] = m43;
    mData[3][3] = m44;
}

void TransformMatrix::SetTranslation(float x, float y, float z)
{
    mData[0][3] = x;
    mData[1][3] = y;
    mData[2][3] = z;
}

void TransformMatrix::Invert()
{
    // Exploit the fact that we are dealing with a rotation matrix + translation component.
    // http://stackoverflow.com/questions/2624422/efficient-4x4-matrix-inverse-affine-transform

    float   tmp;
    float   vals[3];

    // Transpose left-upper 3x3 (rotation) sub-matrix
    tmp = mData[0][1]; mData[0][1] = mData[1][0]; mData[1][0] = tmp;
    tmp = mData[0][2]; mData[0][2] = mData[2][0]; mData[2][0] = tmp;
    tmp = mData[1][2]; mData[1][2] = mData[2][1]; mData[2][1] = tmp;

    // Multiply translation component (last column) by negative inverse of upper-left 3x3.
    for (int i = 0; i < 3; ++i)
    {
        vals[i] = 0.0f;
        for (int j = 0; j < 3; ++j)
        {
            vals[i] += -mData[i][j] * mData[j][3];
        }
    }
    for (int i = 0; i < 3; ++i)
    {
        mData[i][3] = vals[i];
    }
}

TransformMatrix TransformMatrix::RotateX(float rads)
{
    return TransformMatrix(1.0, 0.0, 0.0, 0.0,
        0.0, cos(rads), -sin(rads), 0.0,
        0.0, sin(rads), cos(rads), 0.0,
        0.0, 0.0, 0.0, 1.0);
}

TransformMatrix TransformMatrix::RotateY(float rads)
{
    return TransformMatrix(cos(rads), 0.0, sin(rads), 0.0,
        0.0, 1.0, 0.0, 0.0,
        -sin(rads), 0.0, cos(rads), 0.0,
        0.0, 0.0, 0.0, 1.0);
}

TransformMatrix TransformMatrix::RotateZ(float rads)
{
    return TransformMatrix(cos(rads), -sin(rads), 0.0, 0.0,
        sin(rads), cos(rads), 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0);
}

TransformMatrix TransformMatrix::operator*(const TransformMatrix& rhs)
{
    TransformMatrix result;

    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            float rowCol = 0.0;
            for (int k = 0; k < 4; ++k)
            {
                rowCol += mData[i][k] * rhs.mData[k][j];
            }
            result.mData[i][j] = rowCol;
        }
    }
    return result;
}

Point4 TransformMatrix::operator*(const Point4& v)
{
    const float* pnt = v.Data();
    float   result[4];

    for (int i = 0; i < 4; ++i)
    {
        float rowCol = 0.0;
        for (int k = 0; k < 4; ++k)
        {
            rowCol += mData[i][k] * pnt[k];
        }
        result[i] = rowCol;
    }
    return Point4(result[0], result[1], result[2], result[3]);
}
