from typing import Tuple


def drone_computer(start_time: Tuple[int, int, int], stop_time: int, image_folder: str):

    from djitellopy import Tello
    from time import sleep
    from urllib.request import urlopen
    import cv2

    # The tello.
    tello = Tello()
    tello.connect()
    sleep(1)
    tello.streamon()
    sleep(1)
    tello.takeoff()
    sleep(5)
    tello.move_up(50)
    sleep(1)
    frame_read = tello.get_frame_read()

    # Tello FPS.
    time_between_frames = 1 / 30 - 0.01

    # Wait to start.
    hours, minutes, seconds = urlopen('https://just-the-time.appspot.com/').read().decode().strip()[-8:].split(':')
    sleep(3600 * (start_time[0] - int(hours)) + 60 * (start_time[1] - int(minutes)) + (start_time[2] - int(seconds)))

    # Main loop.
    for i in range(int((60 * stop_time) // (time_between_frames + 0.01))):
        frame = frame_read.frame
        cv2.imwrite(f'{image_folder}/{i}.png', frame)
        cv2.imshow('', frame)
        cv2.waitKey(1)
        tello.send_rc_control(0, 0, 0, 15)
        sleep(time_between_frames)

    # Close.
    tello.streamoff()
    tello.land()


def optitrack_computer(start_time: Tuple[int, int, int], stop_time: int, file_path: str, rigid_body_name: str):

    import natnetclient as natnet
    from time import sleep
    from urllib.request import urlopen

    # The natnet client.
    client = natnet.NatClient(client_ip='127.0.0.1', data_port=1511, comm_port=1510)

    # The rigid body to track.
    rigid_body = client.rigid_bodies[rigid_body_name]

    # The file to save data to.
    data_file = open(file_path, 'w')

    # Tello FPS.
    time_between_frames = 1 / 30

    # Wait to start.
    hours, minutes, seconds = urlopen('http://just-the-time.appspot.com/').read().decode().strip()[-8:].split(':')
    sleep(3600 * (start_time[0] - int(hours)) + 60 * (start_time[1] - int(minutes)) + (start_time[2] - int(seconds)))

    # Main loop.
    for _ in range(int((60 * stop_time) // time_between_frames)):
        pitch, yaw, roll = rigid_body.rotation
        data_file.write(f'{pitch} {yaw} {roll}\n')
        sleep(time_between_frames)

    # Close file.
    data_file.close()


if __name__ == '__main__':
    # drone_computer((0, 0, 0), 1, '')
    # optitrack_computer((0, 0, 0), 1, '', '')
    pass
