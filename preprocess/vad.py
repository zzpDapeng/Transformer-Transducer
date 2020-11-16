import numpy as np


class LtsdVad:
    def __init__(self, win_time_size, order, e0, e1, thre0, thre1, radio, samp_rate, window=None):
        """
        initialize the parameters of lstd vad algorithm
        :param win_time_size: the size of sliding window on signal, that is, the duration time of signal in a window(ms)
        :param window: the window added on the signal, if it is None, a hanning window is provided
        :param order: the order of ltse and ltsd
        :param e0: the min energy threshold corresponding to the quiet condition
        :param e1: the max energy threshold corresponding to the noisy condition
        :param thre0: the ltsd threshold corresponding to the quiet condition
        :param thre1: the ltsd threshold corresponding to the noisy condition
        :param radio: control the noise spectrum updating
        :param samp_rate: the sampling rate of signal
        """
        print("In the init of LtsdVad")
        self.samp_rate = samp_rate
        self.win_time_size = win_time_size
        # convert the "ms" to the "num of samples"
        self.winsize = int(self.win_time_size / 1000 * self.samp_rate)
        self.win_shift_size = self.winsize // 2
        print(self.win_time_size, self.winsize, self.win_shift_size)
        if window is not None:
            self.window = window
        else:
            self.window = np.hanning(self.winsize)
        self.window_num = None
        # store the amplitude values of different frames
        self.amplitudes = {}
        # the average noise amplitude spectrum
        self.avgnoise_amp = None
        self.order = order
        self.e0 = e0
        self.e1 = e1
        self.thre0 = thre0
        self.thre1 = thre1
        self.radio = radio
        self.noise_frame_sum = 0

    def get_frame(self, signal, index):
        start_index = int(index * self.win_shift_size)
        end_index = int(start_index + self.winsize)
        frame = signal[start_index: end_index]
        return frame

    def get_amplitudes_of_frame(self, signal, index):
        if index in self.amplitudes:
            return self.amplitudes[index]
        else:
            frame = self.get_frame(signal, index)
            spec = np.fft.rfft(frame * self.window)
            amp = np.abs(spec)
            self.amplitudes[index] = amp
            return amp

    def ltse(self, signal, index):
        sample_num = len(self.get_amplitudes_of_frame(signal, index))
        max_amp = np.zeros(shape=sample_num)
        for idx in range(index - self.order, index + self.order + 1):
            amp = self.get_amplitudes_of_frame(signal, idx)
            max_amp = np.maximum(max_amp, amp)
        return max_amp

    def ltsd(self, signal, index):
        if index < self.order or index + self.order >= self.window_num:
            return 0
        else:
            return 10.0 * np.log10(np.average(self.ltse(signal, index) ** 2 / self.avgnoise_amp ** 2))

    @staticmethod
    def cal_energy(amps):
        return 10.0 * np.log10((np.average(amps)) ** 2)

    def is_signal(self, signal, index):
        e = self.cal_energy(self.avgnoise_amp)
        print("======")
        print("The average noise energy is %f" % e)
        ltsd = self.ltsd(signal, index)
        print("The ltsd of a signal frame is %f" % ltsd)
        if e < self.e0:
            if ltsd > self.thre0:
                return True
            else:
                return False
        elif e > self.e1:
            if ltsd > self.thre1:
                return True
            else:
                return False
        else:
            thre = ((self.thre0 - self.thre1) / (self.e0 - self.e1)) * e + self.thre0 - (self.thre0 - self.thre1) / \
                   (1 - self.e1 / self.e0)
            print("The new threshold is %f" % thre)
            print("======")
            if ltsd > thre:
                return True
            else:
                return False

    def _compute_noise_avgspectrum(self, nsignal):
        windownum = int(len(nsignal) // self.win_shift_size - 1)
        if self.winsize % 2 == 0:
            nfft = int(self.winsize // 2 + 1)
        else:
            nfft = int(self.winsize + 1) // 2
        avgamp = np.zeros(nfft)
        for l in range(windownum):
            avgamp += self.get_amplitudes_of_frame(signal=nsignal, index=l)
        return avgamp / float(windownum)

    def vad(self, signal, noise=None, samples_num=1600):
        ori_type = type(signal[0])
        print("The data type of signal is %s" % ori_type)
        if ori_type in [np.int8, np.int16, np.int32]:
            signal = np.array(signal).astype(np.int32)
        else:
            signal = np.array(signal).astype(np.float32)
        if noise is not None:
            self.avgnoise_amp = self._compute_noise_avgspectrum(noise)
        else:
            noise = signal[-samples_num:]
            noise = np.array(noise)
            if noise.all() == 0:
                noise = np.random.random_integers(low=1, high=10, size=noise.shape)
            print("The noise corresponding to the final frames is %s" % noise)
            self.avgnoise_amp = self._compute_noise_avgspectrum(noise)
        self.window_num = int((len(signal) - self.winsize) // self.win_shift_size + 1)
        # record the start location and end location of speech in the signal
        speech_pairs = list()
        location_pair = np.empty(shape=2)
        # indicate the previous frame whether is a signal frame
        prev = 0
        for index in range(self.window_num):
            if self.is_signal(signal, index):
                start_location = self.win_shift_size * index
                if prev:
                    speech_pairs[-1][1] = start_location + self.winsize
                else:
                    location_pair[0] = start_location
                    location_pair[1] = start_location + self.winsize
                    speech_pairs.append(location_pair)
                    location_pair = np.empty(shape=2)
                prev = 1
            else:
                self.noise_frame_sum += 1
                if self.noise_frame_sum % 20 == 0:
                    self.update_the_aver_noise_amp_spec(signal, index)
                prev = 0
        speech = np.empty(shape=0)
        for item in speech_pairs:
            speech = np.concatenate([speech, signal[np.int32(item[0]): np.int32(item[1])]])
        return np.array(speech).astype(ori_type)

    def update_the_aver_noise_amp_spec(self, signal, index):
        if self.winsize % 2 == 0:
            nfft = int(self.winsize / 2 + 1)
        else:
            nfft = int((self.winsize + 1) / 2)
        avgamp = np.zeros(shape=nfft)
        for idx in range(index - self.order, index + self.order + 1):
            avgamp += self.get_amplitudes_of_frame(signal, index)
        avgamp = avgamp / np.float32(self.order * 2 + 1)
        self.avgnoise_amp = self.avgnoise_amp * self.radio + avgamp * (1 - self.radio)


if __name__ == "__main__":
    from scipy.io import wavfile

    fs, sig = wavfile.read(filename="E:\gmm_test_experiment\\test.wav")
    print(sig.shape)
    adaptive_lstd = LtsdVad(win_time_size=20, window=None, order=6, e0=30, e1=50, thre0=40.5, thre1=19.5, radio=0.95,
                            samp_rate=fs)
    vad_speech = adaptive_lstd.vad(sig)
    print(vad_speech.shape)
    print(type(vad_speech[0]))
    wavfile.write("E:\gmm_test_experiment\\test_vad.wav", fs, vad_speech)
    from tools.slience_remove import remove_silence

    signal_denosing = remove_silence(fs, sig, perc=0.00125)
    print(signal_denosing.shape)
    wavfile.write("E:\gmm_test_experiment\\test_denosing.wav", fs, signal_denosing)
