import cv2
from ffpyplayer.player import MediaPlayer

video_path  = "video.mp4"

video = cv2.VideoCapture(video_path)

player = MediaPlayer(video_path)
while True:
#     grabbed, frame = video.read()
#     audio_frame, val = player.get_frame()
#     print(audio_frame)
#     print(val)
#     if not grabbed:
#         break
    if cv2.waitKey(28) & 0xFF == ord("q"):
        break
#     cv2.imshow("video", frame)
#     if val != 'eof' and audio_frame is not None:
#         img, t = audio_frame

    # print(img)
    # print(t)