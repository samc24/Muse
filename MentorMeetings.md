# 04/23/2021 Meeting with Ben

**Final Step in Pipeline**
I had trouble with converting OpenCV frames to RTMP. I spoke with Ben about possible solutions that would allow me this conversion. The most efficient solution that came up was FFMPEG. We decided to take each frame and push it through FFMPEG via RTMP to YouTube Live. We also discussed possible issues with audio since audio would be cut out during the live stream. We couldn't come up with a simple enough solution to implement within the remaining time so we decided to skip the audio for now or use an instrumental audio track.

# 04/14/2021 Meeting with Michael

**user flows for web**

  * For this meeitng with Michael, I went over the current user-flows on the website, specifically the steps users should take in order to upload videos
  * I showed him the current designs (mostly finalized)
  * Discussed confusing buttons on editing page, as well as copywriting that can clarify actions that users can take
  * Gave tips on faciliating changes with developers, as well as general work-flow


# 03/26/2021 Meeting with Ben 

**CV for motion detection**

* For our meeting with Ben we primarily went over the progress we've made in virtual panning
* We showed our results of using frame differencing, background generation, and pixel segmentation for calculating the net motion of movement. 
* We discussed using mturk workers for labelling, and discussed what we actually need to label going forward. We talked about labeling the ball for better detection, the basket to detect whether there is a foreign object in the vicinity (indicating a shot attempt), etc.  
* How to determine the start and end of play - based on the median motion line
* Difficulty in distinguishing between steals and poor shot attempts
* Getting it to pan smoothly despite the constant shifting of the median motion line - perhaps only pan when moves from one halfcourt to the other (or quarter)

# 03/12/2021 Meeting with Ben 

**Information on Machine Learning and Object Detection**

* For our meeting with Ben we primarily went over the problems we encountered with object detection.
* We had a few problems with differentiating between the ball and the court since they are the same color.
* We discussed using traditional CV methods for the task of panning the camera. We broke the solution down into two distinct parts, first for the panning of the camera in realtime, and second for advanced video processing for highlight detection, etc. 
* We explored using frame differencing, motion detection, and color filtering. 
* We talked about the coming difficulty in engineering the solution from camera in the facility to our models to youtube. 
* For highlight detection, we discussed the possibility of ball path tracking to determine the outcome of a given play. 
* We talked about how it may not be worth it to do individual player detection and tracking. 

# 03/03/2021 Meeting with Ace 

**High-level project discussion**

* We had a long discussion on the overall goal and outcomes of the project, as well as latter phases
* We discussed how this is generating 'personal basketball cards' for amateur players
* Capitalizing on NFTs to generate personal tokens where the market defines the value of amateur players and their 'cards'
* The relatively small sphere of basketball + tech and how everyone knows each other for the most part - Sloan sports conference is big for this
* Getting in touch with https://www.crossover-india.org/photos/founder-shaun-jayachandran/ and others for international sports clubs
* Analytics is a very difficult and saturated field - have to provide enough value to clubs and players to make this useful. 
