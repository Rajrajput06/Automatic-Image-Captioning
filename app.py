import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import EncoderCNN, DecoderRNN
import pickle

# Load vocabulary
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = EncoderCNN(embed_size=256).to(device)
decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=len(vocab)).to(device)

encoder.load_state_dict(torch.load('./models/encoder-3.pkl', map_location=device))
decoder.load_state_dict(torch.load('./models/decoder-5.pkl', map_location=device))
encoder.eval()
decoder.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

# Caption generation function
def generate_caption(image):
    image = transform(image).unsqueeze(0).to(device)
    features = encoder(image)
    sampled_ids = decoder.sample(features)
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word == '<end>':
            break
        if word != '<start>':
            sampled_caption.append(word)
    return ' '.join(sampled_caption)

# Streamlit UI
st.title("🖼️ Image Caption Generator")
st.write("Upload an image, and the model will generate a caption.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        caption = generate_caption(image)

    st.markdown(f"**Generated Caption:** {caption}")