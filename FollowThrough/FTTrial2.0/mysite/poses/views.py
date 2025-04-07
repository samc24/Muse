from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import time
from os import system
import os
import threading
import mimetypes
import json

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

def upload_view(request): # called when manually uploaded
    return render(request, "poses/upload.html")

def request_view(request): # called by human when request
    return render(request, "poses/request.html")

def folder(request): # list all folders. Heroku deletes uploads folders for some reason
    cwd = os.getcwd()
    files = os.listdir(cwd)
    return HttpResponse('\n'.join([cwd, ';'.join(files)]))

@csrf_exempt
def video_storing(request): # called to store the video and its output
    '''Only supports mp4 now'''
    if request.method == 'POST':
        videoName = str(int(time.time() *1000)) + '.mp4'
        videoPath = 'OpenPose/uploads/' + videoName # added OpenPose/ for demo 12/11/19
        try:
            destination = open(videoPath, 'wb+')
        except FileNotFoundError:
            os.mkdir("uploads")
            destination = open(videoPath, 'wb+')
        for chunk in request.FILES.get('video', False).chunks():
            destination.write(chunk)
        destination.close()
        cvThread = threading.Thread(target=video_analysis, args=(videoName,))
        cvThread.start()    
        return HttpResponse(videoName)

@csrf_exempt
def get_output(request): # when it is called again to see if uploading
    '''Only supports mp4 now'''
    if request.method == 'POST':
            filename = request.POST['path']
            #find status
            with open("finished_videos/" + "progress.json", "r+") as fp:
                    analyzingDict = json.loads(fp.read())
            try:
                    analyzingDict[filename]
            except KeyError:
                    return HttpResponse("The file %s is not in the process of analyzing" % filename)
            else:
                    if analyzingDict[filename] == "in progress":
                            return HttpResponse("The file %s is in the process" % filename)
                    elif analyzingDict[filename] == "done": 
                            videoPath = "finished_videos/" + filename
                            destination = open(videoPath, 'rb+')
                            response = HttpResponse(destination, content_type=mimetypes.guess_type(videoPath)[0])
                            response['Content-Disposition'] = "attachment; filename={0}".format(videoPath)
                            return response

def video_analysis(path): # calls the pose analysis to work in the background by threading
    # system("python OpenPose/OpenPoseVideo.py " + path)
    system("python OpenPose/save_pose_data.py " + path)

