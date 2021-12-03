import numpy as np
import cv2
import time

import signal
import sys
import os
import pathlib
import threading

caps = []
cap_paths = []
frame_number = 0
ports = [1, 2, 3, 4, 5, 6]
cameras = len(ports)
camera_range = range(cameras)
live = True

print(f"\n   session started\n   cameras: {cameras}\n   press ctrl+Z to terminate\n")

def writer(camera):
    _, frame = caps[camera].read()

    try:
        cv2.imwrite(f"{cap_paths[camera]}/{frame_number}.jpg", frame)
    except:
        pass

class Recorder:

    def __init__(self):
        self.make_dirs()

        signal.signal(signal.SIGINT, self.stimulus)
        signal.signal(signal.SIGTSTP, self.signal_handler)

        #self.dims = {"width" : 160, "height" : 120} #4 cameras via USB C
        self.dims = {"width" : 320, "height" : 240} #3 cameras via USB C

        self.target_shape = (480, 640, 3) #camera raw image dimensions
        self.stim = False
        self.initiate_ports()

    def stimulus(self, sig, frame):
        if self.stim == False:
            self.stim_file.write(f"{time.time()}\n")
            self.stim = True
            print("    (!!) stimulation ON")
        else:
            self.stim_file.write(f"{time.time()}\n")
            self.stim = False
            print("    (!!) stimulation OFF")

    def make_dirs(self):
        print("generating directories...")
        curr_path = pathlib.Path(__file__).parent.absolute()
        self.exp_path = f"{curr_path}/data/{time.strftime('%Y%m%d-%H%M%S')}"
        os.mkdir(self.exp_path)

        self.log_path = f"{self.exp_path}/datalog.log"
        self.stim_path = f"{self.exp_path}/stimlog.log"
        self.stim_file = open(self.stim_path, "a+")
        print("directories generated")

    def signal_handler(self, sig, frame):
        global live
        print("\nscript closing")
        live = False

    def initiate_ports(self):
        print("initiating cameras..")
        for port in ports:
            cap_ = cv2.VideoCapture(port)

            if cap_ is not None or cap_.isOpened():
                _, frame = cap_.read()

                if frame.shape == self.target_shape:

                    cap_.set(3, self.dims["width"])
                    cap_.set(4, self.dims["height"])
                    cap_.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                    cap_.set(cv2.CAP_PROP_EXPOSURE, -7.0)
                    caps.append(cap_)

                    cap_path = self.exp_path + f"/cam_{port}"
                    cap_paths.append(cap_path)
                    os.mkdir(cap_path)
                    print(f"camera port {port} initiated")
                else:
                    print(f"turned off camera port {port}")
                    cap_.release()


    def record(self):
        global frame_number

        log_file = open(self.log_path, "a+")

        threads = [0] * cameras

        print("\ncontrols:\nctrl+z = terminate\nctrl+c = stimulus ON/OFF")

        print("recording...")

        while live:
            for camera in camera_range:
                thread_ = threading.Thread(target = writer, args = (camera,))
                threads[camera] = thread_
                thread_.start()

            for thread in threads:
                thread.join()
            timestamp = time.time()
            log_file.write(f"{timestamp}\n")
            frame_number += 1


        print("recording terminated")

        for cap in caps:
            cap.release()

        log_file.close()
        self.stim_file.close()

        print(f"files fully saved\ndir: {self.exp_path}")

        cv2.destroyAllWindows()

        log_file = open(self.log_path, "r")
        lines = log_file.readlines()
        fps = round(len(lines)/(float(lines[-1]) - float(lines[0])), 2)
        print(f"\n   stats:\n   frames: {len(lines)}\n   avg fps: {fps}\n")
        log_file.close()
        sys.exit(0)

r = Recorder()
r.record()
