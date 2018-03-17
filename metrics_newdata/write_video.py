import cv2
import numpy as np

MATCHES_PATH = "./frames_slides_matches/"
imagesToWrite = [
    "0-54-match.jpg",
    "1000-54-match.jpg",
    "2000-54-match.jpg",
    "3000-54-match.jpg",
    "4000-54-match.jpg",
    "5000-54-match.jpg",

    "6000-58-match.jpg",
    "7000-58-match.jpg",
    "8000-58-match.jpg",
    "9000-58-match.jpg",
    "10000-58-match.jpg",   
    "11000-58-match.jpg",
    "12000-58-match.jpg",
    "13000-58-match.jpg",
    "14000-58-match.jpg",
    "15000-58-match.jpg",
    "16000-58-match.jpg",
    "17000-58-match.jpg",
    "18000-58-match.jpg",
    "19000-58-match.jpg",

    "20000-61-match.jpg",
    "21000-61-match.jpg",
    "22000-61-match.jpg",
    "23000-61-match.jpg",
    "24000-61-match.jpg",
    "25000-61-match.jpg",
    "26000-61-match.jpg",
    "27000-61-match.jpg",

    "28000-67-match.jpg",
    "29000-67-match.jpg",
    "30000-67-match.jpg",
    "31000-67-match.jpg",
    "32000-67-match.jpg",
    "33000-67-match.jpg",
    "34000-67-match.jpg",
    "35000-67-match.jpg",
    "36000-67-match.jpg",
    "37000-67-match.jpg",

    "38000-71-match.jpg",
    "39000-71-match.jpg",
    "40000-71-match.jpg",
    "41000-71-match.jpg",

    "42000-72-match.jpg",            
    "43000-72-match.jpg",            
    "44000-72-match.jpg",            
    "45000-72-match.jpg",            
    "46000-72-match.jpg",            
    "47000-72-match.jpg",            
    "48000-72-match.jpg",            
    "49000-72-match.jpg",            
    "50000-72-match.jpg",            
    "51000-72-match.jpg",            
    "52000-72-match.jpg",            
    "53000-72-match.jpg",            
    "54000-72-match.jpg",            
    "55000-72-match.jpg",            
    "56000-72-match.jpg",            
    "57000-72-match.jpg",            
    "58000-72-match.jpg",            
    "59000-72-match.jpg",            
    "60000-72-match.jpg",            
    "61000-72-match.jpg",            
    "62000-72-match.jpg"           
]
writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30,(1304, 405))
for itw in imagesToWrite:
    img = cv2.imread(MATCHES_PATH + itw)
    for i in range(30):
        writer.write(img)
writer.release()