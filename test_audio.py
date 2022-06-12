import Hear

ear = Hear.Hear()

ear.start_listening()

for i in range(100):
    ear.show_current_MFCC()

ear.stop_listening()