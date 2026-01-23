import shutil, os, random, yaml

with open('params.yaml') as f:
    params = yaml.safe_load(f)

N = params["prepare"]["samples_per_class"]
scriptdir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scriptdir)
targetdir = os.path.join(parent_dir, "training_dataset")
initialdir = params["prepare"]["source_path"]

if not os.path.exists(targetdir):
    os.makedirs(targetdir)

for class_name in os.listdir(initialdir):
    class_path = os.path.join(initialdir, class_name)
    if not os.path.isdir(class_path):
        continue
    
    target_class_dir = os.path.join(targetdir, class_name)
    if not os.path.exists(target_class_dir):
        os.makedirs(target_class_dir)
    
    photos_dir = os.path.join(class_path, "all_photos")
    if not os.path.exists(photos_dir):
        continue
    
    all_photos = [f for f in os.listdir(photos_dir) 
                  if os.path.isfile(os.path.join(photos_dir, f))]
    
    if len(all_photos) < N:
        selected_photos = all_photos
    else:
        selected_photos = random.sample(all_photos, N)
    
    for photo in selected_photos:
        src_path = os.path.join(photos_dir, photo)
        dst_path = os.path.join(target_class_dir, photo)
        shutil.copy2(src_path, dst_path)