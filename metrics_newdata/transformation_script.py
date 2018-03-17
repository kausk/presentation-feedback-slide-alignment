import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
 
vidFramePath = "./newdata_vid_frames_sampled_fullframe/png/"


def homography_and_warp(src_fn, dst_fn, out_fn, src_corners, dst_corners): 
    # source: http://www.learnopencv.com/homography-examples-using-opencv-python-c/
    # Read source image.
    im_src = cv2.imread(src_fn)
    # Four corners of the book in source image
    pts_src = np.array(src_corners)
 
 
    # Read destination image.
    im_dst = cv2.imread(dst_fn)
    # Four corners of the book in destination image.
    pts_dst = np.array(dst_corners)
 
    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)
     
    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
    
    cv2.imwrite(out_fn, im_out)
    
def get_gnomonic_hom(center_lat_lon, origin_image, height_width, fov_vert_hor=(60.0, 60.0), out_path=None,
                     plot=False):
    org_height_width, _ = origin_image.shape[:2], origin_image.shape[-1]
    height, width = height_width
    result_image = np.zeros((height, width, 3))

    sphere_radius_lon = width / (2.0 * np.tan(np.radians(fov_vert_hor[1] / 2.0)))
    sphere_radius_lat = height / (2.0 * np.tan(np.radians(fov_vert_hor[0] / 2.0)))

    y, x = np.mgrid[0:height, 0:width]
    x_y_hom = np.column_stack([x.ravel(), y.ravel(), np.ones(len(x.ravel()))])

    K_inv = np.zeros((3, 3))
    K_inv[0, 0] = 1.0/sphere_radius_lon
    K_inv[1, 1] = 1.0/sphere_radius_lat
    K_inv[0, 2] = -width/(2.0*sphere_radius_lon)
    K_inv[1, 2] = -height/(2.0*sphere_radius_lat)
    K_inv[2, 2] = 1.0

    R_lat = np.zeros((3,3))
    R_lat[0,0] = 1.0
    R_lat[1,1] = np.cos(np.radians(-center_lat_lon[0]))
    R_lat[2,2] = R_lat[1,1]
    R_lat[1,2] = -1.0 * np.sin(np.radians(-center_lat_lon[0]))
    R_lat[2,1] = -1.0 * R_lat[1,2]

    R_lon = np.zeros((3,3))
    R_lon[2,2] = 1.0
    R_lon[0,0] = np.cos(np.radians(-center_lat_lon[1]))
    R_lon[1,1] = R_lon[0,0]
    R_lon[0,1] = - np.sin(np.radians(-center_lat_lon[1]))
    R_lon[1,0] = - R_lon[0,1]

    R_full = np.matmul(R_lon, R_lat)

    dot_prod = np.sum(np.matmul(R_full, K_inv).reshape(1,3,3) * x_y_hom.reshape(-1, 1, 3), axis=2)

    sphere_points = dot_prod/np.linalg.norm(dot_prod, axis=1, keepdims=True)

    lat = np.degrees(np.arccos(sphere_points[:, 2]))
    lon = np.degrees(np.arctan2(sphere_points[:, 0], sphere_points[:, 1]))

    lat_lon = np.column_stack([lat, lon])
    lat_lon = np.mod(lat_lon, np.array([180.0, 360.0]))

    org_img_y_x = lat_lon / np.array([180.0, 360.0]) * np.array(org_height_width)
    org_img_y_x = np.clip(org_img_y_x, 0.0, np.array(org_height_width).reshape(1, 2) - 1.0).astype(int)
    org_img_y_x = org_img_y_x.astype(int)

    result_image[x_y_hom[:, 1].astype(int), x_y_hom[:, 0].astype(int), :] = origin_image[org_img_y_x[:, 0],
                                                                    org_img_y_x[:, 1], :]

    if plot:
        fig, ax = plt.subplots(figsize=(16.0, 10.0))
        ax.imshow(result_image.astype('uint8'))
        show(block=False)

    if out_path: plt.imsave(out_path, result_image)

def extract_360_gnomonic(img_path, out_dir, fov_vert_hor=(75,75), size=(300,300)):
# def extract_360_gnomonic(img_path, out_dir, fov_vert_hor=(106, 96), size=(128,128)):
    image = matplotlib.image.imread(img_path).astype(float)[:, :, :3]

    for y in np.arange(0.0, 180.0, np.floor(fov_vert_hor[0]/2.0)):
        for x in np.arange(0.0, 360.0, np.floor(fov_vert_hor[1]/2.0)):
            out_path = os.path.join(out_dir, "%d_%d.jpg"%(y, x))
            get_gnomonic_hom((y, x), image, size, fov_vert_hor=fov_vert_hor, out_path=out_path)


def transform_image(img_name):
    # STEP 1: define the root of your project (root) and the full path to the frame
    # you're trying to convert (src_frame_path)
    root = "/Users/ksk/img/metrics_newdata/newdata_vid_frames_sampled_fullframe/png/"
    src_frame_path = root + img_name

    # STEP 2: select where all of the projections will be saved 
    # (sampled from around the equirectangular projection)  
    proj_out_dir = "/Users/ksk/img/projected_imgs/"

    # optional - mess with these to change the projection width/height 
    # and the field of view of the projections
    proj_out_size = (400,400)
    fov_vert_hor = (70,70) 



    # STEP 4: look in proj_out_dir and find the projection where the slide was
    # centered (e.g., 60_175.jpg) and set the center_lat_lon accordingly, 
    # you may want to adjust a little
    center_lat_lon = (74,185)

    # STEP 5: run this to create the projection that you'll use for the final warping
    proj_out_path = proj_out_dir + "%d_%d-virb.png"%center_lat_lon
    image = matplotlib.image.imread(src_frame_path).astype(float)[:, :, :3]
    get_gnomonic_hom(center_lat_lon, image, proj_out_size, fov_vert_hor, proj_out_path)

    # #STEP 6: Where do you want to save the final image? (hom_out_path)
    hom_out_path = "./new_frames_projected/" + img_name
    print(hom_out_path)
    # STEP 7: set src_corners to be the 4 corners of slide in perspective projection
    # I use Illustrator with the Measure Tool to figure out what the corners are
    ## dimensions from video

    src_corners = [[38,68],[284,83],[298,218],[10,213]] 
    ## src_corners = [[1580,640],[2146,604],[2148,988],[1568,1000]] 

    # STEP 8: where is the filepath of the matching slide, what are the slide's corners?
    # this should be the original slide outputted from powerpoint as an image
    ## dimensions from slide
    dst_fn = "/Users/ksk/img/metrics_newdata/new_slides/3-slide.png"
    dst_corners =  [[0,260],[5000,0],[5000,2800],[0,2800]]

    # STEP 9: run this cell to save the final image to hom_out_path
    homography_and_warp(proj_out_path, dst_fn, hom_out_path, src_corners, dst_corners)


videoFrameNames = [ filename for filename in os.listdir(vidFramePath) if filename.endswith(".png") ]
for v in videoFrameNames:
    transform_image(v)