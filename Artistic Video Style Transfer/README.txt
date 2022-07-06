** USAGE **

1. Test on an input video | test-on-video.py
   -- direction ('up','down','left,'right')  
   -- segmentation ('y','n')
   -- model (Optical flow model file .pth path)
   -- path (Content video frame directory)

2. Motion Transfer | motion_transfer.py
   -- model (Optical flow model file .pth)
   -- path (Content video frame directory)
   -- style_path (style motion video frame directory)
   -- style_model (style transfer model file .pth path)