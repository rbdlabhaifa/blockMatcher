#include <windows.h>
#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "NPTrackingTools.h"

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <opencv2/imgproc/imgproc.hpp>

#include "time.h"
#include "ctello.h"

using namespace cv;

// Local function prototypes
void CheckResult( NPRESULT result );

// Main application
int main( int argc, char* argv[] )
{

    char save_frames_to[] = "";
    char save_csv_to[] = "";
    char project_path[] = "C:\\Users\\fares\\Documents\\OptiTrack\\Motive\\Samples\\markers\\x64\\Debug\\project.ttp";



    printf("== NaturalPoint Tracking Tools API Marker Sample =======---\n");
    printf("== (C) NaturalPoint, Inc.\n\n");

    printf("Initializing NaturalPoint Devices\n");
    TT_Initialize();

    // Do an update to pick up any recently-arrived cameras.
    TT_Update();

    // Load a project file from the executable directory.
    printf( "Loading Project: project.ttp\n\n" );
    CheckResult( TT_LoadProject(project_path) );

    // List all detected cameras.
    printf( "Cameras:\n" );
    for( int i = 0; i < TT_CameraCount(); i++)
    {
        printf( "\t%s\n", TT_CameraName(i) );
    }
    printf("\n");

    // List all defined rigid bodies.
    printf("Rigid Bodies:\n");
    for( int i = 0; i < TT_RigidBodyCount(); i++)
    {
        printf("\t%s\n", TT_RigidBodyName(i));
    }
    printf("\n");

    // URL where the Tello sends its video stream to.
    const char* const TELLO_STREAM_URL{"udp://0.0.0.0:11111"};

    Tello tello{};
    if (!tello.Bind())
    {
        return 0;
    }

    tello.SendCommand("streamon");
    while (!(tello.ReceiveResponse()))
        ;

    VideoCapture capture{TELLO_STREAM_URL, CAP_FFMPEG};

    // Take-off first
    tello.SendCommand("takeoff");
    while (!(tello.ReceiveResponse()))
        ;

    bool busy{false};


    int frameCounter = 0;

    VideoCapture cap(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 960);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
    if (!cap.isOpened()) {
        std::cerr << "BAD" << std::endl;
    }
    else {
        std::cout << "GOOD" << std::endl;
    }

    // waitKey(5000);


    char buff[DTTMSZ];

    // Poll API data until the user hits a keyboard key.
    while( !_kbhit() )
    {
        // waitKey(100);
        if( TT_Update() == NPRESULT_SUCCESS )
        {
            frameCounter++;

            // Update tracking information every 100 frames (for example purposes).
            if( (frameCounter % 100) == 0 )
            {


                int cnt = 0;
                for (int i = 0; i < TT_RigidBodyCount(); i++)
                {
                    if (TT_IsRigidBodyTracked(i))
                    {
                        cnt += TT_RigidBodyMarkerCount(i);
                    }
                }
                if (cnt < 3) {
                    printf("No drone found!\n");
                    continue;
                }

                float   yaw,pitch,roll;
                float   x,y,z;
                float   qx,qy,qz,qw;
                bool    tracked;

                Mat frame;
                cap >> frame;
                // Show what the Tello sees
                resize(frame, frame, Size(), 0.75, 0.75);
                imshow("CTello Stream", frame);
                if (waitKey(1) == 27)
                {
                    break;
                }

                // Listen response
                if (const auto response = tello.ReceiveResponse())
                {
                    std::cout << "Tello: " << *response << std::endl;
                    busy = false;
                }
                
                if (!busy)
                {
                    tello.SendCommand("land");
                    std::cout << "Command: " << command << std::endl;
                    busy = true;
                }

                imwrite(save_frames_to + "/" + std::to_string(frameCounter) + ".png", frame);
                printf( "Frame #%d: (Markers: %d)\n", frameCounter, TT_FrameMarkerCount() );

                for( int i = 0; i < TT_RigidBodyCount(); i++ )
                {
                    std::ofstream rot_rigid_file(save_csv_to + "/" + std::to_string(i) + ".csv", std::ios::app);
                    
                    TT_RigidBodyLocation( i, &x,&y,&z, &qx,&qy,&qz,&qw, &yaw,&pitch,&roll );

                    if( TT_IsRigidBodyTracked( i ) )
                    {
                        printf( "%s: Pos (%.3f, %.3f, %.3f) Orient (%.1f, %.1f, %.1f)\n", TT_RigidBodyName( i ),
                            x, y, z, yaw, pitch, roll );
                        rot_rigid_file << yaw << "," << pitch << "," << roll << "\n";
                    }
                    else
                    {
                        printf( "\t%s: Not Tracked\n", TT_RigidBodyName( i ) );
                    }
                }
            }
        }
        Sleep(2);
    }

    printf( "Shutting down NaturalPoint Tracking Tools\n" );
    CheckResult( TT_Shutdown() );

    printf( "Complete\n" );
    while( !_kbhit() )
    {
        Sleep(20);
    }

    return 0;
}



void CheckResult( NPRESULT result )   //== CheckResult function will display errors and ---
                                      //== exit application after a key is pressed =====---
{
    if( result!= NPRESULT_SUCCESS)
    {
        // Treat all errors as failure conditions.
        printf( "Error: %s\n\n(Press any key to continue)\n", TT_GetResultString(result) );
        std::cout << result << std::endl;

        Sleep(20);
        exit(1);
    }
}
