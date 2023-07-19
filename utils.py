import numpy as np
import torch as th
import random
import os, json
import time
import imageio
import cv2

def seed_init(seed, device):
    th.manual_seed(seed);
    if device != th.device("cpu"):
        with th.cuda.device(device): th.cuda.manual_seed(seed);
    np.random.seed(seed);
    random.seed(seed);
    # th.backends.cudnn.deterministic = True;
    # th.backends.cudnn.benchmark = False;

def hms(x):
    y = int(x);
    return "{}h {:02d}m {:02d}.{:01d}s".format(y // 3600, y // 60 % 60, y % 60, int(x % 1 * 10));

def make_result_dir():
    result_path = "Results";
    result_dirs = os.listdir(result_path);
    result_idx = 0;
    while "Result{}".format(result_idx) in result_dirs: result_idx += 1;
    result_path = "{}/Result{}/".format(result_path, result_idx);
    os.mkdir(result_path);
    os.mkdir(result_path + "Samples");
    os.mkdir(result_path + "Tested");
    os.mkdir(result_path + "Rendered");
    print("Logging on", result_path);
    return result_path;

def load_data_blender(path, data_type, half_res):
    with open(path + "/transforms_{}.json".format(data_type), "r") as f: meta = json.load(f);
    
    imgs, poses, f_nms = [], [], [];
    for frame in meta["frames"]:
        img_fname = path + '/' + frame['file_path'] + ".png";
        imgs.append(imageio.imread(img_fname));
        poses.append(np.array(frame["transform_matrix"]));
        f_nms.append(frame['file_path'].split('/')[-1]);
        
    imgs = np.array(imgs).astype(np.float32) / 255.0;
    poses = np.array(poses).astype(np.float32)[:, :3];
    
    if half_res:
        h, w = imgs.shape[1] // 2, imgs.shape[2] // 2;
        imgs_h = np.zeros((imgs.shape[0], h, w, 4), dtype = np.float32);
        for i, img in enumerate(imgs): imgs_h[i] = cv2.resize(img, (h, w), interpolation = cv2.INTER_AREA);
        imgs = imgs_h;
    
    imgs = imgs[..., :3] * imgs[..., -1:] + (1.0 - imgs[..., -1:]);
    focal = 0.5 * imgs.shape[2] / np.tan(0.5 * float(meta["camera_angle_x"]));
    
    rot_x = lambda phi: th.tensor([[1, 0, 0, 0], [0, np.cos(phi), -np.sin(phi), 0], [0, np.sin(phi), np.cos(phi), 0], [0, 0, 0, 1]], dtype = th.float32);
    rot_y = lambda the: th.tensor([[np.cos(the), 0, -np.sin(the), 0], [0, 1, 0, 0], [np.sin(the), 0, np.cos(the), 0], [0, 0, 0, 1]], dtype = th.float32);
    tsl_z = lambda trs: th.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, trs], [0, 0, 0, 1]], dtype = th.float32);
    lst_m = th.tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype = th.float32);
    
    render_poses = th.stack(
        [lst_m @ rot_y(angle / 180.0 * np.pi) @ rot_x(-30.0 / 180.0 * np.pi) @ tsl_z(4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], dim = 0
    )[:, :3];
    
    return imgs, poses, focal, render_poses, f_nms;

def log(phase, args, result_path):
    if phase == "start":
        hparam, T_start, new_s = args;
        if new_s:
            with open(result_path + "hparam.json", "w") as f: json.dump(hparam, f, indent = 4);
            with open(result_path + "train_log.txt", "w") as f:
                f.write("Training Start ({})\n".format(hms(time.time() - T_start)));
        else:
            with open(result_path + "train_log.txt", "a") as f:
                f.write("New Training Start ({})\n".format(hms(time.time() - T_start)));
    
    elif phase == "eval":
        epoch, lpips, psnr, ssim, smp_imgs, T_eval = args;
        with open(result_path + "train_log.txt", 'a') as f:
            f.write("\tValid set, lpips: {:.4f}, psnr: {:.2f}, ssim: {:.4f}\n"
                    "\tTook {:.2f}s\n".format(lpips, psnr, ssim, time.time() - T_eval));
        for i, img in enumerate(smp_imgs):
            imageio.imwrite(result_path + "Samples/{}_{}.png".format(epoch, i), img);
    
    elif phase == "train":
        epoch, cors_loss, fine_loss, loss, T_start, T_train = args;
        with open(result_path + "train_log.txt", 'a') as f:
            f.write("Epoch {:06d} ({})\tCoarse Loss: {:.6f}\tFine Loss: {:.6f}\tLoss: {:.6f}\tTook {:.2f}s\n".format(
                    epoch, hms(time.time() - T_start), cors_loss, fine_loss, loss, time.time() - T_train));
    
    elif phase == "finish":
        history, T_start = args;
        best_lpips_idx, best_psnr_idx, best_ssim_idx = np.argmin(history["lpips"]), np.argmax(history["psnr"]), np.argmax(history["ssim"]);
        
        with open(result_path + "train_log. txt", 'a') as f:
            f.write("\nTraining Done. ({})\n"
                    "\tBest lpips epoch: {} (lpips: {:.5f}\tpsnr: {:.3f}\tssim: {:.5f})\n"
                    "\tBest psnr  epoch: {} (lpips: {:.5f}\tpsnr: {:.3f}\tssim: {:.5f})\n"
                    "\tBest ssim  epoch: {} (lpips: {:.5f}\tpsnr: {:.3f}\tssim: {:.5f})\n".format(hms(time.time() - T_start),
                    history["epoch"][best_lpips_idx], history["lpips"][best_lpips_idx], history["psnr"][best_lpips_idx], history["ssim"][best_lpips_idx],
                    history["epoch"][best_psnr_idx],  history["lpips"][best_psnr_idx],  history["psnr"][best_psnr_idx],  history["ssim"][best_psnr_idx],
                    history["epoch"][best_ssim_idx],  history["lpips"][best_ssim_idx],  history["psnr"][best_ssim_idx],  history["ssim"][best_ssim_idx]));


def test_log(phase, args, result_path):
    if phase == "start":
        hparam, T_start = args;
        with open(result_path + "test_log.txt", "w") as f:
            f.write("Test Start ({})\n".format(hms(time.time() - T_start)));
    
    elif phase == "write":
        im_nm, smp_img, lpips, psnr, ssim, T_start, T_eval = args;
        with open(result_path + "test_log.txt", "a") as f:
            f.write("{} ({})\tlpips: {:.5f}\tpsnr: {:.3f}\tssim: {:.5f}\tTook {:.2f}s\n".format(
                    im_nm, hms(time.time() - T_start), lpips, psnr, ssim, time.time() - T_eval));
        imageio.imwrite(result_path + "Tested/{}.png".format(im_nm), smp_img);
    
    elif phase == "finish":
        T_start, history = args;
        with open(result_path + "test_log.txt", "a") as f:
            f.write("\nTest Done. ({})\n\tlpips: {:.5f}\tpsnr: {:.3f}\tssim: {:.5f}\n".format(
                    hms(time.time() - T_start), np.mean(history["lpips"]), np.mean(history["psnr"]), np.mean(history["ssim"])));


def render_log(phase, args, result_path):
    if phase == "start":
        hparam, T_start = args;
        with open(result_path + "render_log.txt", "w") as f:
            f.write("Rendering Start ({})\n".format(hms(time.time() - T_start)));
    
    elif phase == "write":
        im_nm, smp_img, T_start, T_eval = args;
        with open(result_path + "render_log.txt", "a") as f:
            f.write("{}\t{} is saved.\tTook {:.2f}s\n".format(hms(time.time() - T_start), im_nm, time.time() - T_eval));
        imageio.imwrite(result_path + "Rendered/{}.png".format(im_nm), smp_img);
    
    elif phase == "finish":
        T_start = args;
        writer = imageio.get_writer(result_path + 'rendered.mp4', fps = 24, format = "FFMPEG");
        for i in range(40): writer.append_data(imageio.imread(result_path + "Rendered/r_{}.png".format(i)));
        writer.close();
        with open(result_path + "render_log.txt", "a") as f:
           f.write("\nRendering Done. ({})\n".format(hms(time.time() - T_start)));
