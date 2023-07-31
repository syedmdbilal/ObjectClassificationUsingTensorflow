from django.shortcuts import render
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from tensorflow import keras

from . import forms
from .models import UserImageModel

# Create your views here.
def home(request):
    print("HI")
    if request.method == "POST":
        form = forms.UserImageForm(files=request.FILES)
        if form.is_valid():
            print('HIFORM')
            form.save()
        obj = form.instance
        #('obj',obj)

        result1 = UserImageModel.objects.latest('id')
        models = keras.models.load_model('E:/Project/CODE/Final Deploy/app/LeNet.h5')
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open("E:/Project/CODE/Final Deploy/media/" + str(result1)).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        classes = ['COMPUTER MOUSE', 'KEYBOARD', 'KEYS', 'LAPTOP', 'PHONE', 'WATCH']
        prediction = models.predict(data)
        idd = np.argmax(prediction)
        a = (classes[idd])
        return render(request, 'app/index.html',{'form':form,'obj':obj,'predict':a})
    else:
        form = forms.UserImageForm()
    return render(request, 'app/index.html',{'form':form})


