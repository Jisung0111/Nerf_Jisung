import time
import numpy as np
import torch as th
import json
import utils
import argparse
from model import Nerf
import pickle

def main(hparam):
    T_start = time.time();
    device = th.device(hparam["gpu"]);
    utils.seed_init(hparam["seed"], device);
    
    train_data = utils.load_data_blender(hparam["data_path"], "train", True);
    valid_data = utils.load_data_blender(hparam["data_path"], "val", False);
    
    if hparam["chkpnt_path"] is None:
        result_path = utils.make_result_dir();
        hparam["chkpnt_path"] = result_path;
        history = {
            "lpips"     : [],
            "psnr"      : [],
            "ssim"      : [],
            "epoch"     : [],
            "cors_loss" : [],
            "fine_loss" : [],
            "loss"      : []
        };
        utils.log("start", [hparam, T_start, True], result_path);
        
    else:
        result_path = hparam["chkpnt_path"];
        with open(result_path + "history.pkl", "rb") as f:
            history = pickle.load(f);
        utils.log("start", [hparam, T_start, False], result_path);
    
    epoch = len(history["loss"]);
    model = Nerf(hparam, device, epoch);
    
    if epoch == 0:
        T_eval = time.time();
        lpips, psnr, ssim, smp_imgs = model.evaluate(valid_data);
        
        history["epoch"].append(epoch); history["psnr"].append(psnr);
        history["lpips"].append(lpips); history["ssim"].append(ssim);
        utils.log("eval", [epoch, lpips, psnr, ssim, smp_imgs, T_eval], result_path);
    
    b_lpips, b_psnr, b_ssim = np.min(history["lpips"]), np.max(history["psnr"]), np.max(history["ssim"]);
    
    for _ in range(hparam["epoch"]):
        T_train = time.time();
        if device != th.device("cpu"):
            with th.cuda.device(device): th.cuda.empty_cache();
        
        epoch, cors_loss, fine_loss, loss = model.run_epoch(train_data);
        
        history["cors_loss"].append(cors_loss); history["fine_loss"].append(fine_loss);
        history["loss"].append(loss);
        utils.log("train", [epoch, cors_loss, fine_loss, loss, T_start, T_train], result_path);
        
        if epoch % hparam["eval_period"] == 0:
            T_eval = time.time();
            if device != th.device("cpu"):
                with th.cuda.device(device): th.cuda.empty_cache();
            
            lpips, psnr, ssim, smp_imgs = model.evaluate(valid_data);
            
            model.save(result_path + "recent_model.pth");
            if b_lpips > lpips: b_lpips = lpips; model.save(result_path + "best_lpips.pth");
            if b_psnr < psnr:   b_psnr = psnr;   model.save(result_path + "best_psnr.pth");
            if b_ssim < ssim:   b_ssim = ssim;   model.save(result_path + "best_ssim.pth");
            
            history["epoch"].append(epoch); history["psnr"].append(psnr);
            history["lpips"].append(lpips); history["ssim"].append(ssim);
            utils.log("eval", [epoch, lpips, psnr, ssim, smp_imgs, T_eval], result_path);
        
        with open(result_path + "history.pkl", "wb") as f: pickle.dump(history, f);
    
    utils.log("finish", [history, T_start], result_path);


if __name__ == "__main__":
    hparam = {
        "seed"           : 1,
        "gpu"            : "cuda:0",           # default is cuda:0
        "data_path"      : "Data/nerf_synthetic/lego",
        
        "batch_size"     : 1,
        "epoch"          : 3000,
        "num_iter"       : 300000,
        
        "pos_dim"        : 10,
        "viw_dim"        : 4,
        "init_lr"        : 0.0005,
        "end_lr"         : 0.00005,
        
        "eval_period"    : 10,            # evaluation period (unit: epoch)
        "eval_num_smp"   : 5,
        
        "num_pixels"     : 4096,
        "num_coarse"     : 64,
        "num_fine"       : 128,
        
        "t_near"         : 2.0,
        "t_far"          : 6.0,
        
        "chkpnt_path"    : None,
        "best_model"     : "recent_model.pth"
    };
    
    parser = argparse.ArgumentParser();
    parser.add_argument('--hparam');
    file_loc = parser.parse_args();
    
    if file_loc.hparam is not None:
        with open(file_loc.hparam) as json_file:
            hparam.update(json.load(json_file));
    
    main(hparam);