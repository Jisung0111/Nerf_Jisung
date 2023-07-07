import time
import torch as th
import json
import utils
import argparse
from model import Nerf
import pickle

def main(hparam, test_opt):
    T_start = time.time();
    device = th.device(hparam["gpu"]);
    utils.seed_init(hparam["seed"], device);
    
    imgs, poses, focal, render_poses, nms = utils.load_data_blender(hparam["data_path"], "test", False);
    
    result_path = hparam["chkpnt_path"];
    model = Nerf(hparam, device);
    
    if test_opt:
        utils.test_log("start", [hparam, T_start], result_path);
        history = {"lpips": [], "psnr": [], "ssim": [], "name": []};
        
        for img, pose, im_nm in zip(imgs, poses, nms):
            T_eval = time.time();
            lpips, psnr, ssim, smp_img = model.test(img, pose, focal);
            
            history["psnr"].append(psnr); history["lpips"].append(lpips);
            history["ssim"].append(ssim); history["name"].append(im_nm);
            utils.test_log("write", [im_nm, smp_img, lpips, psnr, ssim, T_start, T_eval], result_path);
            
            with open(result_path + "history_test.pkl", "wb") as f: pickle.dump(history, f);
        
        utils.test_log("finish", [T_start, history], result_path);
    
    else:
        h_, w_ = imgs.shape[1:3];
        utils.render_log("start", [hparam, T_start], result_path);
        
        for i, pose in enumerate(render_poses):
            T_eval = time.time();
            smp_img = model.make_image(pose, h_, w_, focal);
            utils.render_log("write", ["r_" + str(i), smp_img, T_start, T_eval], result_path);
        
        utils.render_log("finish", T_start, result_path);


if __name__ == "__main__":
    parser = argparse.ArgumentParser();
    parser.add_argument('--hparam');
    parser.add_argument('--test');
    file_loc = parser.parse_args();
    
    with open(file_loc.hparam) as f: hparam = json.load(f);
    if file_loc.test is None: raise ValueError("Enter whether you test by 0 or 1");
    
    main(hparam, int(file_loc.test));
