import numpy as np
import cv2
from scipy import signal
import matplotlib.pyplot as plt
from constants import moving_average, width, height, reject_outliers
import os

class HR:
    def __init__(self, dir:str, ID:str, ON:bool = True, skip:bool = False, offset:int = 15):#1: 15 #2:25
        self.ID = ID
        self.dir = dir
        self.offset = offset

        self.show_selection = False#True
        self.diff_means=[]
        self.sums = []
        self.best_idx_=[]

        self.save_ = False#True
        if self.save_:
            save_dir = f"{self.dir}/vids/heartrate_sample.avi"
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            self.out = cv2.VideoWriter(save_dir, fourcc, 32.0, ((width*3), (height*2)), True)
            print(save_dir)


        self.c1s=[]

        if ON:

            if self.check_if_exists() and skip:
                #print("loading pre-existing heartrate data")
                self.load = lambda x,y,z,q:None
                self.compute = lambda:self.load_heartrate()
            else:
                #print("generating heartrate data")
                self.load = self.load_
                self.compute = self.compute_
        else:
            self.load = lambda x,y,z,q:None
            self.compute = lambda:None

    def check_if_exists(self):
        return os.path.isfile(f"{self.dir}/vids/heartrate.npy")


    def load_(self, cam:int, marker:np.ndarray, score:np.ndarray, times:np.ndarray):
        return

        marker = np.flip(marker.copy(), axis = 1).astype(int) #y,x

        self.times = times - times[0]
        cap = cv2.VideoCapture(f"{self.dir}/vids/{self.ID}_cam_{cam}.mp4")

        diff_means = np.zeros(marker.shape[0], dtype=np.float32)
        sums=np.zeros(marker.shape[0], dtype=int)
        best_idx=np.zeros(marker.shape[0], dtype=int)

        offset = self.offset
        bounds = (np.clip(marker[:,0] - offset, a_min=0,a_max=width), np.clip(marker[:,0] + offset, a_min=0,a_max=height), np.clip(marker[:,1] - offset,a_min=0,a_max=width), np.clip(marker[:,1] + offset, a_min=0,a_max=width))
        a,b,c,d = np.asarray([bound for bound in bounds]).astype(int)

        rots = offset
        resolution = offset * 2

        contours = np.array([[0, 0], [resolution, resolution], [resolution, 0],[0,0]], dtype='int32')
        canvas = np.zeros((resolution,) * 2)
        contours_ = np.zeros((*contours.shape, rots))

        rotation_masks = np.zeros((*canvas.shape, rots),dtype=np.uint8)
        offset_masks = np.zeros((2, *canvas.shape, rots, resolution), dtype=np.uint8) #normal, inverted

        steps = offset * 2
        rot_step_size = (offset * 2)//rots

        for n in np.arange(rots):
            contours[0][1] += rot_step_size
            contours[1][1] -= rot_step_size
            contours[0][1] = min(resolution, contours[0][1])
            contours[1][1] = max(0, contours[1][1])
            #contours_[:, :, n] = contours.copy()
            rotation_masks[:,:,n] = cv2.fillConvexPoly(canvas.copy(), points = contours, color = 1)

            for f, factor in enumerate([1, -1]):
                for i in np.arange(resolution):
                    cont = contours.copy()
                    cont[0][1] -= factor
                    cont[1][1] -= factor
                    offset_masks[f,:, :, n, i] = cv2.fillConvexPoly(canvas.copy(), points = cont, color = 1)

        rot_stds = np.zeros((2, rots), dtype=np.float64)
        axes = (0,1,3)
        offset_axes = (0,1,2)
        inverted_rotationmasks = 1 - rotation_masks
        for n in np.arange(marker.shape[0]):
            frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
            crop = frame[a[n]:b[n], c[n]:d[n]]

            normal_mask = crop[rotation_masks]
            inverted_mask = crop[inverted_rotationmasks]

            rot_stds[0] = np.std(normal_mask, axis = axes)
            rot_stds[1] = np.std(inverted_mask, axis = axes)

            best = np.argmin(rot_stds, axis = 1)
            #(2, 30) (2, 2) (2,) (2,)
            #print(rot_stds.shape, rot_stds[:, best].shape, best.shape, np.diag(rot_stds[:, best]).shape)
            best_dir = np.argmin(np.diag(rot_stds[:, best]))
            best_contour = best[best_dir]
            #print(best.shape,best_dir.shape) #(2,) ()
            #print(best_contour.shape, best_contour)

            #print(best, best_dir, best_contour)
            #print(offset_masks[0].shape)
            #print(offset_masks[best_dir][:, :, best_contour, :].shape)
            mask = crop[offset_masks[best_dir,:, :, best_contour, :]]
            #print(mask.shape)

            std = np.std(mask, axis = offset_axes)
            best_idx[n] = np.argmin(std)
            bi=best_idx[n - 1]
            diff_means[n] = std[bi]
            sums[n] = np.sum(mask, axis = offset_axes)[bi]

            #print(n)

        self.diff_means.append(diff_means)
        self.sums.append(sums)

        self.best_idx_.append(best_idx)


    def compute_(self):
        L = 200
        if True:
            self.best_idx_ = np.load(f"{self.dir}/vids/best_idx.npy", allow_pickle=True)
            self.diff_means = np.load(f"{self.dir}/vids/stds.npy", allow_pickle=True)
            self.times = np.load(f"{self.dir}/vids/times.npy", allow_pickle=True)
            self.sums = np.load(f"{self.dir}/vids/sums.npy", allow_pickle=True)

        else:
            np.save(f"{self.dir}/vids/sums2.npy",self.sums)
            np.save(f"{self.dir}/vids/best_idx2.npy",self.best_idx_)
            np.save(f"{self.dir}/vids/stds2.npy",self.diff_means)
            np.save(f"{self.dir}/vids/times.npy",self.times)

        comb_bpms=[]

        #print(self.diff_means.shape, self.sums.shape,self.best_idx_.shape)

        #return None
        #diff_mean *= self.best_idx_[n]
        #print(self.sums.shape, self.diff_means)
        #print((self.diff_means == 0).sum())
        div = np.divide(1, self.diff_means, out=np.ones_like(self.diff_means), where=self.diff_means!=0)
        #print(div.shape)

        combined = np.sqrt(self.sums * np.exp((self.best_idx_/(self.offset * 2)))) #*div#*(self.best_idx_/(self.offset*2))# # * (div)* self.diff_means
        #print(combined.shape)
        #plt.plot(combined.T )
        #plt.show()

        #N = 1
        #combined = np.asarray([moving_average(combined[n,:], N) for n in np.arange(6)])
        #plt.plot(combined.T )
        #plt.show()
        windows = np.lib.stride_tricks.sliding_window_view(combined, L, axis = 1)#[::L//10]

        time_window = np.lib.stride_tricks.sliding_window_view(self.times, L)#[::L//10]
        #print(windows.shape)
        bpms = np.zeros((6, windows.shape[1]), dtype=np.float32)

        for n in np.arange(windows.shape[1]):

            time_s = time_window[n][0]
            time_e = time_window[n][-1]
            #print("KO")
            self.fps = float(L) / (time_e - time_s)#calculate HR using a true fps of processor of the computer, not the fps the camera provide
            even_times = np.linspace(time_s, time_e, L*2)
            #print(windows[n].shape, self.times[windows_n[n][0]:windows_n[n][-1]].shape, self.times.shape, even_times.shape)
            processed = signal.detrend(windows[:,n,:], axis = 1)#detrend the signal to avoid interference of light change
            #print(processed.shape)
            interpolated = np.asarray([np.interp(even_times, time_window[n], processed[i,:]) for i in np.arange(6)]) #interpolation by 1
            interpolated = np.hamming(L*2) * interpolated#make the signal become more periodic (advoid spectral leakage)
            #print(interpolated.shape)
            #norm = (interpolated - np.mean(interpolated))/np.std(interpolated)#normalization
            norm = interpolated.T/np.linalg.norm(interpolated, axis = 1)
            #print(norm.shape)
            raw = np.fft.rfft(norm * 30, axis = 0)#do real fft with the normalization multiplied by 10
            raw = np.real(raw)

            #f, y =signal.welch(interpolated)

            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)
            freqs = 60. * self.freqs

            self.fft = np.abs(raw)**2#get amplitude spectrum

            idx = np.where((freqs > 100) & (freqs < 600))[0]#the range of frequency that HR is supposed to be within
            #print(self.fft.shape, )
            #print(self.fft.shape, idx.shape)
            pruned = self.fft[idx,:]
            pfreq = freqs[idx]

            self.freqs = pfreq
            self.fft = pruned

            #plt.plot(pruned)
            #plt.show()
            #print("kkmkm",pruned.shape)
            idx2 = np.argmax(pruned, axis = 0)#max in the range can be HR
            #print(self.freqs.shape, idx2.shape, bpms.shape, self.freqs[idx2].shape, idx2)
            #print(self.freqs[idx2])
            #print(bpms[:, n])
            bpms[:, n] = self.freqs[idx2]
            #print("K")
            #print(n)
        #comb_bpms.append(bpms)
        #print(bpms.shape)

        comb_bpms = np.asarray(bpms)

        #print(comb_bpms.shape)
        comb_bpms = np.asarray([reject_outliers(bpm, comb_bpms) for bpm in comb_bpms])

        comb_bpms = np.nanmedian(np.asarray(comb_bpms), axis=0)

        self.save(comb_bpms)
        mavg = comb_bpms#moving_average(comb_bpms, L//20)
        return mavg

    def save(self, data):
        np.save(f"{self.dir}/vids/heartrate.npy", data)
        #print("heart-rate data saved")

    def load_heartrate(self):
        #print("loading heart rate data")
        return np.load(f"{self.dir}/vids/heartrate.npy", allow_pickle = True)
