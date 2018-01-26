#
# To use this script, just copy/paste it into the directory containing the screenshots
# and run with ./make_video. It will output "video.mp4" in the same directory.
# The screenshots are assumed to be in increasing order with 5-digit filenames, e.g.
# 00001.png, 00002.png, etc...
#
# The -r <number> is the frame rate (frames per second). You'll need to adjust that as needed.
#
# 0.04 dt = 25
# 0.02 dt = 50
#

ffmpeg -y -f image2 -r 25 -i %05d.png -vcodec libx264 -preset medium -b 10M -pix_fmt yuv420p -vf scale="-1:1080,pad=1920:ih:(ow-iw)/2" video.mp4
#ffmpeg -y -f image2 -r 14 -i %05d.png -vcodec libx264 -preset medium -b 10M -pix_fmt yuv420p ../video.mp4
#ffmpeg -y -f image2 -r 50 -i %05d.png -vcodec libx264 -preset medium -b 10M -pix_fmt yuv420p ../video.mp4
#ffmpeg -y -f image2 -r 2 -i %05d.png -vcodec libx264 -preset medium -b 10M -pix_fmt yuv420p ../video.mp4

