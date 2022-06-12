# OVERLAP FOR AUDIO ====================================================================================================
# def read(self):
#     # sends out the collected waveform up to the point it is called in an MFCC format
#     waveform_snapshot = self.waveform
#     self.waveform = []
#     #
#     # if self.previous_waveform:
#     #     overlap = self.extract_overlap(waveform_snapshot)
#     #     current_waveform += self.previous_waveform[-overlap:]
#
#     current_waveform += waveform_snapshot
#     self.previous_waveform = waveform_snapshot + self.waveform
#
#     current_mfcc = self.convert_to_MFCC(np.array(waveform_snapshot, dtype=np.float32))
#     return current_mfcc
#
#
# def stop_listening(self):
#     # closes the process and termintates the audio stream
#     self.listening = False
#     self.thread.join()
#     self.stream.stop_stream()
#     self.stream.close()
#     self.p.terminate()
#
#
# def convert_to_MFCC(self, waveform):
#     # converts the current waveform to MFCC
#     # MFCC is an image format
#     # MFCC is a feature extraction of a sound wave that mimics the information collected by the human ear
#     return (librosa.feature.mfcc(y=waveform, sr=self.sample_rate, n_mfcc=self.n_mfcc))
#
#
# def extract_overlap(self, waveform):
#     # The overlap should add to the main signal to make the output a power of 2
#     # find the closest power of 2 that is greater than the current amount of samples in current waveform(len) = ceil
#     # subtract the number len from ceil == overlap = ceil - len
#     # collect overlap from the previous waveform and add it to the back of current waveform
#
#     number_of_data = len(waveform)
#     print(number_of_data)
#     log_of_len = math.log(number_of_data, 2)
#     closest_power_of_two = math.floor(log_of_len) + 1
#     overlap = (2 ** closest_power_of_two) - number_of_data
#     print(number_of_data, closest_power_of_two, overlap)
#     return overlap