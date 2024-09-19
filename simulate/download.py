import pandas as pd
import os
import librosa
import soundfile as sf

dataset_folder = '../dataset/audioset'
split = 'eval'
tsv = f'{dataset_folder}/audioset_{split}_strong.tsv'
video_folder = f'{dataset_folder}/audioset_{split}_strong'
image_folder = f'{dataset_folder}/audioset_{split}_strong_images'
audio_folder = f'{dataset_folder}/audioset_{split}_strong_audios'

os.makedirs(video_folder, exist_ok=True)
os.makedirs(image_folder, exist_ok=True)
os.makedirs(audio_folder, exist_ok=True)

existing_files = os.listdir(video_folder)
df = pd.read_csv(tsv, sep='\t')
def download_segment(args):
    segment_id, video_folder, existing_files = args
    ytid, starttime = segment_id.rsplit('_', 1)
    
    if ytid + '.mp4' in existing_files:
        print(f'{ytid} already exists, skipping')
        return
    
    starttime = int(starttime) / 1000
    endtime = starttime + 10
    converted_starttime = f'{int(starttime // 3600):02d}:{int((starttime % 3600) // 60):02d}:{starttime % 60:06.3f}'
    converted_endtime = f'{int(endtime // 3600):02d}:{int((endtime % 3600) // 60):02d}:{endtime % 60:06.3f}'
    
    youtube_url = f'https://www.youtube.com/watch?v={ytid}'
    os.system(f'yt-dlp {youtube_url} -P {video_folder} -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" '
              f'-o "{segment_id}.%(ext)s" --download-sections "*{converted_starttime}-{converted_endtime}"')
MODE = 'preprocess' # 'download' or 'process'
if MODE == 'download':
    from multiprocessing import Pool, Manager
    # get the unique segment_id
    unique_segment_id = df['segment_id'].unique()

    args = [(segment_id, video_folder, existing_files) for segment_id in unique_segment_id]
    # remove the repea
    # print(args)
    
    # Use a Manager to share existing_files between processes
    with Manager() as manager:
        with Pool(processes=8) as pool:
            pool.map(download_segment, args)
elif MODE == 'preprocess': # process the downloaded files
    for existing_file in existing_files:
        # existing_file = 's9d-2nhuJCQ_30000.mp4'
        plain_fname = existing_file.split('.')[0]
        if os.path.exists(f'{audio_folder}/{plain_fname}.flac') and os.path.exists(f'{image_folder}/{plain_fname}.jpg'):
            pass
            #print(f'{plain_fname} already exists, skipping')
        else:
            # print(os.path.exists(f'{audio_folder}/{plain_fname}.flac'), os.path.exists(f'{image_folder}/{plain_fname}.jpg'))
            # continue
            # try:
            # ffmpeg extract the audio with flac format
            os.system(f'ffmpeg -i {video_folder}/{existing_file} -vn -acodec flac {audio_folder}/{plain_fname}.flac -y')
            # crop the middle image of the video
            os.system(f'ffmpeg -i {video_folder}/{existing_file} -vf "select=eq(n\\,5)" -vsync vfr {image_folder}/{plain_fname}.jpg -y')
            # except Exception as e:
            #     print(e)
            #     # remove the generated audio and image
            #     os.remove(f'{audio_folder}/{plain_fname}.flac')
            #     os.remove(f'{image_folder}/{plain_fname}.jpg')
        # break
elif MODE == 'embedding':
    # use openai-clip to embed the image and save the embedding
    from PIL import Image
    import torch
    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device="cuda")

    # image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
    # text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    # with torch.no_grad():
    #     image_features = model.encode_image(image)
    #     text_features = model.encode_text(text)
        
    #     logits_per_image, logits_per_text = model(image, text)
    #     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    for existing_file in existing_files:
        plain_fname = existing_file.split('.')[0]
        image_fname = f'{image_folder}/{plain_fname}.jpg'
        print(image_fname)
        image = preprocess(Image.open(image_fname)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            print(image_features.shape)
        break
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
