"""
URL configuration for backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from django.contrib import admin
from django.urls import path
from .views import *

schema_view = get_schema_view(
   openapi.Info(
      title="Audio Source Separation",
      default_version='v1',
      description="API for instrument extraction",
      terms_of_service="https://www.google.com/policies/terms/",
      contact=openapi.Contact(email="contact@yourapi.local"),
      license=openapi.License(name="BSD License"),
   ),
   public=True,
   permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('api/separate_audio/', separate_audio, name='separate_audio'),
    path('api/process_audio_with_effects/', process_audio_with_effects, name='process_audio_with_effects'),
    path('api/create_export/', create_export, name='create_export'),
    path('api/get_image/', get_image, name='get_image'),
    path('api/upload_and_add_project/', upload_and_add_project, name='upload_and_add_project'),
    path('api/delete_project/', delete_project, name='delete_project'),
    path('api/get_projects_by_credential/', get_projects_by_credential, name='get_projects_by_credential'),
    path('api/get_project_file/', get_project_file, name='get_project_file'),
]


