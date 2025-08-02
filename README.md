## Temporary Notes

installing mp42mcap dependencies:
* libavutil-dev 
* libavcodec-dev 
* libavformat-dev 
* libswscale-dev 
* libswresample-dev 
* libavfilter-dev
* libavdevice-dev

Error: "This video contains B-frames or reordered frames (PTS=0, DTS=-1024). Please re-encode the video without B-frames using: ffmpeg -i <input> -c:v libx264 -bf 0 output.mp4"

