import threading

import cv2
from typing import List
import numpy as np
from djitellopy import Tello
from threading import Thread


def capture(tello: Tello, save_to: str = 'image.png'):
    frame = tello.get_frame_read().frame
    cv2.imwrite(save_to, frame)
    print(f'Captured to {save_to} .')


def rotate(tello: Tello, theta: str = '0'):
    value = int(theta)
    if theta < 0:
        tello.rotate_counter_clockwise(-theta)
        print(f'rotated counter-clockwise by {-theta} degrees.')
    else:
        tello.rotate_clockwise(theta)
        print(f'rotated clockwise by {theta} degrees.')


def up_down(tello: Tello, value: str):
    value = int(value)
    if value > 0:
        tello.move_up(value)
    else:
        tello.move_down(-value)


def forward_backward(tello: Tello, value: str):
    value = int(value)
    if value > 0:
        tello.move_forward(value)
    else:
        tello.move_back(-value)


def left_right(tello: Tello, value: str):
    value = int(value)
    if value > 0:
        tello.move_left(value)
    else:
        tello.move_right(-value)


def move(tello: Tello, value: str = '20'):
    value = int(value)
    image_number = 0
    print('You can now control the drone freely with your keyboard.')
    print('Keys: w, a, s, d, e, q, r, f.')
    print('Press esc to stop and space to capture image.')
    while True:
        cv2.imshow('', tello.get_frame_read().frame)
        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break
        elif key == ord(' '):
            capture(tello, f'/home/rani/PycharmProjects/blockMatcher/Dictionary/{image_number}.png')
            image_number += 1
        elif key == ord('w'):
            tello.move_forward(value)
        elif key == ord('s'):
            tello.move_back(value)
        elif key == ord('a'):
            tello.move_left(value)
        elif key == ord('d'):
            tello.move_right(value)
        elif key == ord('e'):
            tello.rotate_clockwise(value)
        elif key == ord('q'):
            tello.rotate_counter_clockwise(value)
        elif key == ord('r'):
            tello.move_up(value)
        elif key == ord('f'):
            tello.move_down(value)


def process_command(tello: Tello, command: str):
    allowed_commands = {
        'cap': capture,
        'rot': rotate,
        'up-down': up_down,
        'forward-backward': forward_backward,
        'left-right': left_right,
        'move': move,
    }
    split_command = ['']
    for i, character in enumerate(command):
        if character == ' ':
            if i != 0 and command[i - 1] == '\\':
                split_command[len(split_command) - 1] += ' '
            else:
                split_command.append('')
        else:
            split_command[len(split_command) - 1] += character
    if split_command[0] not in allowed_commands.keys():
        print(f'Command {split_command[0]} is not a valid command!')
        print('Allowed commands are:', allowed_commands.keys())
        return
    threading.Thread(target=allowed_commands[split_command[0]], args=(tello, *split_command[1:])).start()


def main():
    tello = Tello()
    tello.connect()
    tello.takeoff()
    tello.streamon()
    while True:
        try:
            command = input('enter command: ').strip().lower()
            if command == 'stop':
                break
            process_command(tello, command)
        except Exception as e:
            print('Error!')
            print(e)
    tello.streamoff()
    tello.land()


if __name__ == '__main__':
    main()
