# Penggunaan
# python3 social_distance_detector.py --input vidio.mp4
# python3 social_distance_detector.py --input vidio.mp4 --output vidio.avi

# import packages yang diperlukan
from pyimagesearch import social_distancing_config as config
from pyimagesearch.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
from cv2 import cv2
import multiprocessing
import time
from playsound import playsound

# Buat argumen, parse, dan parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# Memuat label kelas COCO tempat model YOLO kami dilatih
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# mendapatkan jalur ke YOLO weights dan model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# Memuat detektor objek YOLO kami yang dilatih pada dataset COCO (80 kelas)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Cek jika menggunakan GPU
if config.USE_GPU:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Tentukan hanya nama layer * output * yang kita butuhkan dari YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
# menginisialisasi video stream dan menunjuk ke file video keluaran
print("[INFO] accessing video stream...")

# Pakai ini jika live streaming kamera
# vs = cv2.VideoCapture(args["input"] if args["input"] else 0)  
# Pakai ini jika pakai vidio training 
vs = cv2.VideoCapture('SD3.mp4')
writer = None

# loop over the frames dari video stream
while True:
	# baca frame selanjutnya dari file tersebut
	(grabbed, frame) = vs.read()
	
	# jika frame tidak diambil, berarti kita telah mencapai akhir aliran / berhenti
	if not grabbed:
		break

	# Resize/atur ukuran frame dan deteksi orang (hanya orang) di sini
	frame = imutils.resize(frame, width=700)
	results = detect_people(frame, net, ln,
		personIdx=LABELS.index("person"))

	# menginisialisasi kumpulan indeks yang melanggar/violate 
	# jarak physical distancing minimum
	violate = set()

	# pastikan ada * setidaknya * dua orang yang terdeteksi (diperlukan 
	# untuk menghitung peta jarak berpasangan)
	if len(results) >= 2:
		# ekstrak semua centroids dari hasil dan hitung jarak 
		# Euclidean antara semua pasang centroids
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")

		# loop over the upper triangular of the distance matrix
		# loop di atas segitiga atas dari matriks jarak
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				# periksa untuk melihat apakah jarak antara dua pasangan 
				# centroids kurang dari jumlah piksel yang dikonfigurasi 
				if D[i, j] < config.MIN_DISTANCE:
					# perbarui kumpulan pelanggaran/violation kami dengan indeks 
					# pasangan sentroid
					violate.add(i)
					violate.add(j)

	# loop over the results
	for (i, (prob, bbox, centroid)) in enumerate(results):
		# inisialisasi suara
		p = multiprocessing.Process(target=playsound, args=("0001.mp3",))
		# ekstrak kotak pembatas dan koordinat pusat, 
		# lalu inisialisasi warna annotation
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)

		# jika objek i masuk ke daerah pelanggaran maka update warna
		if i in violate:
			color = (0, 0, 255)
		# jika pelanggaran lebih dari 5 detik, suara berbunyi
		if i in violate:
			time.sleep(5)
			p.start()

		# Menggambar kotak untuk orang dan lingkaran untuk pusat orang
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.circle(frame, (cX, cY), 5, color, 1)

	# Menuliskan INCLUDE dan total nomer orang yang melanggar physical distancing
	text = "Pelanggaran Physical Distancing : {}".format(len(violate))
	cv2.putText(frame, text, (10, frame.shape[0] - 40),
		cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255),2)
	text = "INCLUDE (Intelligent Camera System for Physical Distancing)"
	cv2.putText(frame, text, (10, frame.shape[0] - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.70, (200, 0, 0), 2)

	# periksa untuk melihat apakah bingkai keluaran harus ditampilkan ke kami
	# Layar
	if args["display"] > 0:
		# tampilkan nama layar
		cv2.imshow("INCLUDE (Intelligent Camera System for Physical Distancing)", frame)
		key = cv2.waitKey(1) & 0xFF

		# Jika menekan 'q' maka keluar dari loop
		if key == ord("q"):
			break

	# jika jalur file video keluaran telah disediakan dan penulis video 
	# belum diinisialisasi, lakukan sekarang
	if args["output"] != "" and writer is None:
		# initialize video yang ditulis
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)

	# Jika tidak ada vidio penulis, akan menuliskan frame ke output vidio file
	if writer is not None:
		writer.write(frame)