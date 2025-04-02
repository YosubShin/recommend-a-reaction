import cv2
import os.path as osp
from insightface.app import FaceAnalysis


def get_image(image_file, to_rgb=False):

    if not osp.exists(image_file):
        raise FileNotFoundError(f'{image_file} not found')
    img = cv2.imread(image_file)
    if to_rgb:
        img = img[:, :, ::-1]
    return img


app = FaceAnalysis(name='antelopev2')  # or use 'buffalo_l' for more speed
app.prepare(ctx_id=-1, det_size=(160, 160))  # or -1 for CPU

image_file = '/Users/yosub/co/recommend-a-reaction/output/asd/jajw-8Bj_Ys/Scene-607/track_0/track_0_frame_7.jpg'
image = get_image(image_file)
faces = app.get(image, max_num=2)

for face in faces:
    print("Face attributes:")
    for attr_name in dir(face):
        # Skip private attributes and methods
        if attr_name.startswith('_') or callable(getattr(face, attr_name)):
            continue

        attr = getattr(face, attr_name)

        # Check if it's a numpy array
        if hasattr(attr, 'shape') and len(attr.shape) > 1:
            print(f"  {attr_name}: shape {attr.shape}")
        elif attr_name == 'embedding' or attr_name == 'normed_embedding':
            print(f"  {attr_name}: length {len(attr)}")
        else:
            print(f"  {attr_name}: {attr}")
