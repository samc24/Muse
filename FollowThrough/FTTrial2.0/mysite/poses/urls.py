from django.urls import path

from . import views
# determines what is called when clicked

urlpatterns = [
    path('', views.index, name='index'),
    path('vs', views.video_storing, name='video_storing'),
    path('uv', views.upload_view, name='upload_view'),
    path('go', views.get_output, name='get_output'),
    path('rv', views.request_view, name='request_view'),
    path('folder', views.folder, name="folder")
]
