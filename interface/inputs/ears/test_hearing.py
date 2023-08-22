import hear as hear

ear = hear.Hear()

ear.start_listening()

for i in range(100):
    ear.show_current_MFCC()
    ear.save_current_MFCC()
    print(ear.current_mfcc_image.shape)

ear.stop_listening()