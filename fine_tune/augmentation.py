import cv2
import os
import albumentations as A
import shutil

# patchs
img_dir = "data/images/train"
lbl_dir = "data/labels/train"
out_img_dir = "data/images/train_augmented"
out_lbl_dir = "data/labels/train_augmented"

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_lbl_dir, exist_ok=True)

# transform
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.7),
    A.RandomGamma(p=0.5),
    A.RandomShadow(p=0.4),
    A.MotionBlur(p=0.3),
])


for filename in os.listdir(img_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(img_dir, filename)
        label_path = os.path.join(lbl_dir, filename.replace('.jpg', '.txt').replace('.png', '.txt'))


        img = cv2.imread(img_path)
        if img is None:
            continue
        with open(label_path, 'r') as f:
            lines = f.read().strip().split('\n')
            bboxes = []
            class_labels = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_labels.append(int(parts[0]))
                    bboxes.append([float(x) for x in parts[1:]])

        # augmentation
        if bboxes:
            transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
            aug_img = transformed['image']
            aug_boxes = transformed['bboxes']
            aug_labels = transformed['class_labels']

            # save augmented image & label
            out_img_name = f"aug_{filename}"
            out_lbl_name = f"aug_{filename.replace('.jpg', '.txt').replace('.png', '.txt')}"

            cv2.imwrite(os.path.join(out_img_dir, out_img_name), aug_img)

            with open(os.path.join(out_lbl_dir, out_lbl_name), 'w') as f:
                for cls, box in zip(aug_labels, aug_boxes):
                    x, y, w, h = box
                    f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
