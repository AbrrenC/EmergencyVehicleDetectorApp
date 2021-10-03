# import libraries
import streamlit as st
from PIL import Image,ImageFilter
import torchvision.transforms as transforms
from torchvision import *
from torch import *
from modelclass import *

# set the background color and title
st.write('''<style>
            body{
            text-align:center;
            background-color: #ACDDDE;
            }
            </style>''', unsafe_allow_html=True)

st.title('Emergency Vehicle Detector')

transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                                 ])

#loading the model
loaded_densenet169 = Densenet169()
loaded_densenet169.load_state_dict(torch.load('models/densenet169.pt',map_location=torch.device('cpu')))
loaded_densenet169.eval()

st.text('training model loaded using densenet169')

file_type = 'jpg'

uploaded_file = st.file_uploader("Upload your file",type = file_type)

if uploaded_file != None:

    image = Image.open(uploaded_file)

    image = image.filter(ImageFilter.MinFilter(3))

    st.image(image)

    image = transform(image).view(1,3,224,224)
    # image = transform(image).view(transform(image).shape[0], -1)

    pred  = loaded_densenet169.forward(image)
    proba,idx = torch.max(torch.sigmoid(pred),dim = 1)

    proba = proba.detach().numpy()[0]
    idx = idx.numpy()[0]    


    if idx == 1:
        st.text('Emergency Vehicle')

    else:
        st.text('Non_Emergency Vehicle')

    st.text(f'class {idx}')
    st.write('confidence {:0.3f}'.format(float(proba)))