from django.db import models


class aids(models.Model):
    age = models.IntegerField()
    sex = models.IntegerField()
    bmi = models.IntegerField()
    children = models.IntegerField()
    smoker = models.IntegerField()
    region = models.IntegerField()
    prediction = models.IntegerField()
