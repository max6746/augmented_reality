import pyglet
import time
vid_path='video.mp4' # Name of the video
# window=pyglet.window.Window()
player = pyglet.media.Player()
# source = pyglet.media.StreamingSource()
MediaLoad = pyglet.media.load(vid_path)
player.on_eos()
# player.loop = True
player.queue(MediaLoad)
player.play()

# @window.event
# def on_draw():
#     if player.source and player.source.video_format:
#         player.get_texture().blit(50,50)

# time.sleep(5)
# player.queue(MediaLoad)
# player.play()
pyglet.app.run()