from Audio import stream_audio
from Redacted.Image import stream_video
from time import sleep
import concurrent.futures

def stream_inputs(stream_time):
    with concurrent.futures.ThreadPoolExecutor(5) as pool:
        thread1 = pool.submit(stream_audio, stream_time = stream_time)
        thread2 = pool.submit(stream_video, stream_time = stream_time)

    audio = thread1.result()
    images = thread2.result()
    return audio, images

def mind():
    stream_time = 1
    #sense environment
    with concurrent.futures.ThreadPoolExecutor(5) as pool:
        thread1 = pool.submit(stream_audio, stream_time=stream_time)
        thread2 = pool.submit(stream_video, stream_time=stream_time)

    #process and act previous senses
    print("now processing and acting on input")
    sleep(5) #replace with actual nerual network agent
    print("done processing")

    #when done processing, add the senses that came in while processing as the new inputs
    audio = thread1.result()
    images = thread2.result()
    print("done")

while True:
    mind()