import cv2 
import os 


output_file = './CK_plus.csv'
img_Path = "./CK+/cohn-kanade-images"
label_Path = "./CK+/Emotion"

output = open(output_file, 'w')
output.write('pixels,emotion,usage\n')

# Directory of all the images
object_Dirs = os.listdir(img_Path)
try:
	object_Dirs.remove('.DS_Store')
except:
	pass

# Directory of all the labels
label_Dirs = os.listdir(label_Path)
try:
	label_Dirs.remove('.DS_Store')
except:
	pass


# Go through each object's (people) folder
for object_img_path, object_label_path in zip(object_Dirs, label_Dirs):

	# Each object has a few folders corresponding to each emotions (not all objects have all emotions)
	object_emo_dirs = os.listdir(os.path.join(img_Path, object_img_path))
	try:
		object_emo_dirs.remove('.DS_Store')
	except:
		pass

	objects_labels_dirs = os.listdir(os.path.join(label_Path, object_label_path))
	try:
		objects_labels_dirs.remove('.DS_Store')
	except:
		pass

	# Go through each emotion folder under each object
	for emo_img_path, emo_label_path in zip(object_emo_dirs, objects_labels_dirs):

		emo_img_files = os.listdir(os.path.join(img_Path, object_img_path, emo_img_path))
		emo_label_files = os.listdir(os.path.join(label_Path, object_label_path, emo_label_path))

		emo_img_files.sort(reverse=True)

		if len(emo_img_files)>0 and len(emo_label_files)>0:
			img = ' '.join(cv2.imread(os.path.join(img_Path, object_img_path, emo_img_path, emo_img_files[0]),0).flatten().astype('str'))
			label = open(os.path.join(label_Path, object_label_path, emo_label_path, emo_label_files[0]), 'r').read().strip()[0]

			output.write(img+','+label+',training\n')

output.close()









