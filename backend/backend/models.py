# models.py

from django.db import models

class Project(models.Model):
    credential = models.CharField(max_length=255)
    project_name = models.CharField(max_length=255)
    project_path = models.CharField(max_length=255)

    def __str__(self):
        return self.project_name

