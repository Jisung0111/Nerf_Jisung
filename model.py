import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from neural_net import MLP
import torchmetrics
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, th.Tensor):
            param.data = param.data.to(device);
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device);
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, th.Tensor):
                    subparam.data = subparam.data.to(device);
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device);

class Nerf:
    def __init__(self, hparam, device, epoch = 0):
        self.batch_size, self.device, self.epoch = hparam["batch_size"], device, epoch;
        self.t_near, self.t_far, self.eval_num_smp = hparam["t_near"], hparam["t_far"], hparam["eval_num_smp"];
        self.num_pixels, self.num_coarse, self.num_fine = hparam["num_pixels"], hparam["num_coarse"], hparam["num_fine"];
        
        self.cors_net = MLP(hparam["pos_dim"], hparam["viw_dim"]);
        self.fine_net = MLP(hparam["pos_dim"], hparam["viw_dim"]);
        
        self.optim = th.optim.Adam(list(self.cors_net.parameters()) + list(self.fine_net.parameters()), lr = hparam["init_lr"]);
        self.scdlr = th.optim.lr_scheduler.ExponentialLR(self.optim, pow(hparam["end_lr"] / hparam["init_lr"], 1 / hparam["num_iter"]));
        
        if hparam["chkpnt_path"] is not None:
            self.load(hparam["chkpnt_path"] + hparam["best_model"]);
        
        if device is not th.device("cpu"):
            self.cors_net.cuda(device); self.fine_net.cuda(device); optimizer_to(self.optim, device);
        
        self.lossf = nn.MSELoss();
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type = 'vgg', normalize = True).to(device);
        self.psnr = torchmetrics.PeakSignalNoiseRatio().to(device);
        self.ssim = torchmetrics.StructuralSimilarityIndexMeasure().to(device);

    def run_epoch(self, data):
        imgs, poses, focal, render_poses, nms = data;
        cors_losses, fine_losses, losses = [], [], [];
        
        self.cors_net.train();
        self.fine_net.train();
        
        for img, pose in zip(imgs, poses):
            h_, w_ = img.shape[:2];
            f_x, f_y, c_x, c_y = focal, focal, w_ / 2, h_ / 2;

            rig = th.from_numpy(img).to(self.device).reshape(-1, 3);
            
            if self.epoch < 5:
                c_i, c_j = (h_ - 1) // 2, (w_ - 1) // 2;
                c_is, c_js = th.arange(c_i - c_i // 2, c_i + c_i // 2), th.arange(c_j - c_j // 2, c_j + c_j // 2);
                c_idx = th.cartesian_prod(c_is, c_js);
                c_idx = c_idx[:, 0] * w_ + c_idx[:, 1];
                pixel_idx = c_idx[th.randperm(len(c_idx))[:self.num_pixels]];
            else:
                pixel_idx = th.tensor(np.random.choice(rig.shape[0], size = [self.num_pixels], replace = False));
            
            pixel_idx, pose_ = pixel_idx.to(self.device), th.from_numpy(pose).to(self.device);
            
            t_ = th.linspace(self.t_near, self.t_far, self.num_coarse + 1, device = self.device)[:-1].unsqueeze(0).repeat(self.num_pixels, 1);
            t_ = t_ + ((self.t_far - self.t_near) / self.num_coarse) * th.rand(t_.shape, device = self.device);
            dt = th.cat([t_[:, 1:] - t_[:, :-1], th.full((self.num_pixels, 1), 1e10, device = self.device)], dim = -1);
            
            x_, y_, z_ = (pixel_idx % w_ - c_x) / f_x, (c_y - pixel_idx // w_) / f_y, -th.ones_like(pixel_idx);
            o_, d_ = pose_[:3, 3].unsqueeze(0), th.sum(th.stack([x_, y_, z_], dim = -1).unsqueeze(1) * pose_[:3, :3], dim = -1);
            d_ = (d_ / th.norm(d_, dim = -1, keepdim = True)).unsqueeze(1);
            # o_, d_, t_, dt: (1, 3), (P, 1, 3), (P, S), (P, S)
            
            sigc, radc = self.cors_net(o_.unsqueeze(1) + d_.repeat(1, t_.shape[1], 1) * t_.unsqueeze(2), d_);
            v_c = -sigc.view((dt.shape[0], -1)) * dt; cv_c = th.cumsum(v_c, dim = -1); cv_c = th.roll(cv_c, 1, -1); cv_c[:, 0] = 0.0;
            wgtc = th.exp(cv_c) * (1.0 - th.exp(v_c));
            rgbc = th.sum(wgtc.unsqueeze(2) * radc, dim = 1);
            # sigc, radc, wgtc, rgbc: (P, S, 1), (P, S, 3), (P, S), (P, 3)
            
            cors_loss = self.lossf(rgbc, rig[pixel_idx]);
            
            t_m, w_m = 0.5 * (t_[:, 1:] + t_[:, :-1]), wgtc.detach()[:, 1:-1] + 1e-8;
            cdf = th.cat([th.zeros_like(w_m[:, :1]), th.cumsum(w_m / th.sum(w_m, dim = -1, keepdim = True), dim = -1)], dim = -1);
            cdf_ys = th.rand((self.num_pixels, self.num_fine), device = self.device);
            # cdf, cdf_ys: (P, S - 1), (P, F)
            
            c_idx = th.searchsorted(cdf, cdf_ys, right = True);
            l_, u_ = th.max(th.zeros_like(c_idx), c_idx - 1), th.min(th.tensor(cdf.shape[-1] - 1), c_idx);
            # l_, u_: (P, F), (P, F)
            
            c_l, t_l = th.gather(cdf, 1, l_), th.gather(t_m, 1, l_);
            q_ = th.gather(cdf, 1, u_) - c_l; q_ = th.where(q_ < 1e-5, th.ones_like(q_), q_);
            t_ = th.sort(th.cat([t_, t_l + (cdf_ys - c_l) * (th.gather(t_m, 1, u_) - t_l) / q_], dim = -1), dim = -1)[0];
            dt = th.cat([t_[:, 1:] - t_[:, :-1], th.full((t_.shape[0], 1), 1e10, device = self.device)], dim = -1);
            # c_l, t_l, q_, t_, dt: (P, F), (P, F), (P, F), (P, F + S), (P, F + S)
            
            sigf, radf = self.fine_net(o_.unsqueeze(1) + d_.repeat(1, t_.shape[1], 1) * t_.unsqueeze(2), d_);
            v_f = -sigf.view((dt.shape[0], -1)) * dt; cv_f = th.cumsum(v_f, dim = -1); cv_f = th.roll(cv_f, 1, -1); cv_f[:, 0] = 0.0;
            wgtf = th.exp(cv_f) * (1.0 - th.exp(v_f));
            rgbf = th.sum(wgtf.unsqueeze(2) * radf, dim = 1);
            # wgtf, rgbf: (P, F + S), (P, 3)
            
            fine_loss = self.lossf(rgbf, rig[pixel_idx]);
            
            loss = cors_loss + fine_loss;
            
            self.optim.zero_grad();
            loss.backward();
            self.optim.step();
            
            cors_losses.append(cors_loss.detach().item());
            fine_losses.append(fine_loss.detach().item());
            losses.append(loss.detach().item());
        
        cors_loss = np.mean(cors_losses);
        fine_loss = np.mean(fine_losses);
        loss = np.mean(losses);
        self.epoch += 1;
        
        return self.epoch, cors_loss, fine_loss, loss;
    
    @th.no_grad()
    def evaluate(self, data):
        imgs, poses, focal, render_poses, nms = data;
        lpips, psnr, ssim, smp_imgs = [], [], [], [];
        
        self.cors_net.eval();
        self.fine_net.eval();
        
        for data_idx, (img, pose) in enumerate(zip(imgs, poses)):
            h_, w_ = img.shape[:2];
            f_x, f_y, c_x, c_y = focal, focal, w_ / 2, h_ / 2;

            pose_, rgbs = th.from_numpy(pose).to(self.device), [];
            pixel_itr = list(range(0, h_ * w_, 6 * self.num_pixels)) + [h_ * w_];
            
            for i in range(len(pixel_itr) - 1):
                if self.device != th.device("cpu"):
                    with th.cuda.device(self.device): th.cuda.empty_cache();
                px_idx, num_pixels = th.arange(pixel_itr[i], pixel_itr[i + 1], device = self.device), pixel_itr[i + 1] - pixel_itr[i];
                
                t_ = th.linspace(self.t_near, self.t_far, self.num_coarse + 1, device = self.device)[:-1].unsqueeze(0).repeat(num_pixels, 1);
                t_ = t_ + ((self.t_far - self.t_near) / self.num_coarse) * th.rand(t_.shape, device = self.device);
                dt = th.cat([t_[:, 1:] - t_[:, :-1], th.full((num_pixels, 1), 1e10, device = self.device)], dim = -1);
                
                x_, y_, z_ = (px_idx % w_ - c_x) / f_x, (c_y - px_idx // w_) / f_y, -th.ones_like(px_idx);
                o_, d_ = pose_[:3, 3].unsqueeze(0), th.sum(th.stack([x_, y_, z_], dim = -1).unsqueeze(1) * pose_[:3, :3], dim = -1);
                d_ = (d_ / th.norm(d_, dim = -1, keepdim = True)).unsqueeze(1);
                # o_, d_, t_, dt: (1, 3), (P, 1, 3), (P, S), (P, S)
                
                sig, rad = self.cors_net(o_.unsqueeze(1) + d_.repeat(1, t_.shape[1], 1) * t_.unsqueeze(2), d_);
                # sig: (P, S, 1)
                v_ = -sig.view((num_pixels, -1)) * dt; cv_ = th.cumsum(v_, dim = -1); cv_ = th.roll(cv_, 1, -1); cv_[:, 0] = 0.0;
                wgt = th.exp(cv_) * (1.0 - th.exp(v_));
                # sig, rad, wgt: (P, S, 1), (P, S, 3), (P, S)
                
                t_m, w_m = 0.5 * (t_[:, 1:] + t_[:, :-1]), wgt[:, 1:-1] + 1e-8;
                cdf = th.cat([th.zeros_like(w_m[:, :1]), th.cumsum(w_m / th.sum(w_m, dim = -1, keepdim = True), dim = -1)], dim = -1);
                cdf_ys = th.rand((num_pixels, self.num_fine), device = self.device);
                # w_m, cdf, cdf_ys: (P, S - 2), (P, S - 1), (P, F)
                
                c_idx = th.searchsorted(cdf, cdf_ys, right = True);
                l_, u_ = th.max(th.zeros_like(c_idx), c_idx - 1), th.min(th.tensor(cdf.shape[-1] - 1), c_idx);
                # l_, u_: (P, F), (P, F)
                
                c_l, t_l = th.gather(cdf, 1, l_), th.gather(t_m, 1, l_);
                q_ = th.gather(cdf, 1, u_) - c_l; q_ = th.where(q_ < 1e-5, th.ones_like(q_), q_);
                t_ = th.sort(th.cat([t_, t_l + (cdf_ys - c_l) * (th.gather(t_m, 1, u_) - t_l) / q_], dim = -1), dim = -1)[0];
                dt = th.cat([t_[:, 1:] - t_[:, :-1], th.full((t_.shape[0], 1), 1e10, device = self.device)], dim = -1);
                # c_l, t_l, q_, t_, dt: (P, F), (P, F), (P, F), (P, F + S), (P, F + S)
                
                sig, rad = self.fine_net(o_.unsqueeze(1) + d_.repeat(1, t_.shape[1], 1) * t_.unsqueeze(2), d_);
                v_ = -sig.view((dt.shape[0], -1)) * dt; cv_ = th.cumsum(v_, dim = -1); cv_ = th.roll(cv_, 1, -1); cv_[:, 0] = 0.0;
                wgt = th.exp(cv_) * (1.0 - th.exp(v_));
                rgb = th.sum(wgt.unsqueeze(2) * rad, dim = 1);
                # wgt, rgb: (P, F + S), (P, 3)
            
                rgbs.append(rgb);
            
            rgb = th.cat(rgbs, dim = 0).reshape((1, h_, w_, -1)).permute(0, 3, 1, 2).clamp(0.0, 1.0);
            rig = th.from_numpy(img).to(self.device).reshape((1, h_, w_, -1)).permute(0, 3, 1, 2);
            # rgb, rig: (1, 3, H, W)
            
            lpips.append(self.lpips(rgb, rig).item());
            psnr.append(self.psnr(rgb, rig).item());
            ssim.append(self.ssim(rgb, rig).item());
            smp_imgs.append((th.cat((rgb, rig), dim = 3).reshape((3, h_, -1)).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8));
            
            if data_idx + 1 == self.eval_num_smp: break;
            
        return np.mean(lpips), np.mean(psnr), np.mean(ssim), smp_imgs;
    
    @th.no_grad()
    def make_image(self, pose, h_, w_, focal):
        f_x, f_y, c_x, c_y = focal, focal, w_ / 2, h_ / 2;
        
        self.cors_net.eval();
        self.fine_net.eval();

        pose_, rgbs = pose.to(self.device), [];
        pixel_itr = list(range(0, h_ * w_, 6 * self.num_pixels)) + [h_ * w_];
        
        for i in range(len(pixel_itr) - 1):
            if self.device != th.device("cpu"):
                with th.cuda.device(self.device): th.cuda.empty_cache();
            px_idx, num_pixels = th.arange(pixel_itr[i], pixel_itr[i + 1], device = self.device), pixel_itr[i + 1] - pixel_itr[i];
            
            t_ = th.linspace(self.t_near, self.t_far, self.num_coarse + 1, device = self.device)[:-1].unsqueeze(0).repeat(num_pixels, 1);
            t_ = t_ + ((self.t_far - self.t_near) / self.num_coarse) * th.rand(t_.shape, device = self.device);
            dt = th.cat([t_[:, 1:] - t_[:, :-1], th.full((num_pixels, 1), 1e10, device = self.device)], dim = -1);
            
            x_, y_, z_ = (px_idx % w_ - c_x) / f_x, (c_y - px_idx // w_) / f_y, -th.ones_like(px_idx);
            o_, d_ = pose_[:3, 3].unsqueeze(0), th.sum(th.stack([x_, y_, z_], dim = -1).unsqueeze(1) * pose_[:3, :3], dim = -1);
            d_ = (d_ / th.norm(d_, dim = -1, keepdim = True)).unsqueeze(1);
            # o_, d_, t_, dt: (1, 3), (P, 1, 3), (P, S), (P, S)
            
            sig, rad = self.cors_net(o_.unsqueeze(1) + d_.repeat(1, t_.shape[1], 1) * t_.unsqueeze(2), d_);
            # sig: (P, S, 1)
            v_ = -sig.view((num_pixels, -1)) * dt; cv_ = th.cumsum(v_, dim = -1); cv_ = th.roll(cv_, 1, -1); cv_[:, 0] = 0.0;
            wgt = th.exp(cv_) * (1.0 - th.exp(v_));
            # sig, rad, wgt: (P, S, 1), (P, S, 3), (P, S)
            
            t_m, w_m = 0.5 * (t_[:, 1:] + t_[:, :-1]), wgt[:, 1:-1] + 1e-8;
            cdf = th.cat([th.zeros_like(w_m[:, :1]), th.cumsum(w_m / th.sum(w_m, dim = -1, keepdim = True), dim = -1)], dim = -1);
            cdf_ys = th.rand((num_pixels, self.num_fine), device = self.device);
            # w_m, cdf, cdf_ys: (P, S - 2), (P, S - 1), (P, F)
            
            c_idx = th.searchsorted(cdf, cdf_ys, right = True);
            l_, u_ = th.max(th.zeros_like(c_idx), c_idx - 1), th.min(th.tensor(cdf.shape[-1] - 1), c_idx);
            # l_, u_: (P, F), (P, F)
            
            c_l, t_l = th.gather(cdf, 1, l_), th.gather(t_m, 1, l_);
            q_ = th.gather(cdf, 1, u_) - c_l; q_ = th.where(q_ < 1e-5, th.ones_like(q_), q_);
            t_ = th.sort(th.cat([t_, t_l + (cdf_ys - c_l) * (th.gather(t_m, 1, u_) - t_l) / q_], dim = -1), dim = -1)[0];
            dt = th.cat([t_[:, 1:] - t_[:, :-1], th.full((t_.shape[0], 1), 1e10, device = self.device)], dim = -1);
            # c_l, t_l, q_, t_, dt: (P, F), (P, F), (P, F), (P, F + S), (P, F + S)
            
            sig, rad = self.fine_net(o_.unsqueeze(1) + d_.repeat(1, t_.shape[1], 1) * t_.unsqueeze(2), d_);
            v_ = -sig.view((dt.shape[0], -1)) * dt; cv_ = th.cumsum(v_, dim = -1); cv_ = th.roll(cv_, 1, -1); cv_[:, 0] = 0.0;
            wgt = th.exp(cv_) * (1.0 - th.exp(v_));
            rgb = th.sum(wgt.unsqueeze(2) * rad, dim = 1);
            # wgt, rgb: (P, F + S), (P, 3)
        
            rgbs.append(rgb);
        
        rgb = th.cat(rgbs, dim = 0).reshape((1, h_, w_, -1)).permute(0, 3, 1, 2).clamp(0.0, 1.0);
        # rgb: (1, 3, H, W)
        
        smp_img = (rgb.reshape((3, h_, -1)).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8);
            
        return smp_img;
    
    @th.no_grad()
    def test(self, img, pose, focal):
        self.cors_net.eval();
        self.fine_net.eval();
        
        h_, w_ = img.shape[:2];
        f_x, f_y, c_x, c_y = focal, focal, w_ / 2, h_ / 2;

        pose_, rgbs = th.from_numpy(pose).to(self.device), [];
        pixel_itr = list(range(0, h_ * w_, 6 * self.num_pixels)) + [h_ * w_];
        
        for i in range(len(pixel_itr) - 1):
            if self.device != th.device("cpu"):
                with th.cuda.device(self.device): th.cuda.empty_cache();
            px_idx, num_pixels = th.arange(pixel_itr[i], pixel_itr[i + 1], device = self.device), pixel_itr[i + 1] - pixel_itr[i];
            
            t_ = th.linspace(self.t_near, self.t_far, self.num_coarse + 1, device = self.device)[:-1].unsqueeze(0).repeat(num_pixels, 1);
            t_ = t_ + ((self.t_far - self.t_near) / self.num_coarse) * th.rand(t_.shape, device = self.device);
            dt = th.cat([t_[:, 1:] - t_[:, :-1], th.full((num_pixels, 1), 1e10, device = self.device)], dim = -1);
            
            x_, y_, z_ = (px_idx % w_ - c_x) / f_x, (c_y - px_idx // w_) / f_y, -th.ones_like(px_idx);
            o_, d_ = pose_[:3, 3].unsqueeze(0), th.sum(th.stack([x_, y_, z_], dim = -1).unsqueeze(1) * pose_[:3, :3], dim = -1);
            d_ = (d_ / th.norm(d_, dim = -1, keepdim = True)).unsqueeze(1);
            # o_, d_, t_, dt: (1, 3), (P, 1, 3), (P, S), (P, S)
            
            sig, rad = self.cors_net(o_.unsqueeze(1) + d_.repeat(1, t_.shape[1], 1) * t_.unsqueeze(2), d_);
            # sig: (P, S, 1)
            v_ = -sig.view((num_pixels, -1)) * dt; cv_ = th.cumsum(v_, dim = -1); cv_ = th.roll(cv_, 1, -1); cv_[:, 0] = 0.0;
            wgt = th.exp(cv_) * (1.0 - th.exp(v_));
            # sig, rad, wgt: (P, S, 1), (P, S, 3), (P, S)
            
            t_m, w_m = 0.5 * (t_[:, 1:] + t_[:, :-1]), wgt[:, 1:-1] + 1e-8;
            cdf = th.cat([th.zeros_like(w_m[:, :1]), th.cumsum(w_m / th.sum(w_m, dim = -1, keepdim = True), dim = -1)], dim = -1);
            cdf_ys = th.rand((num_pixels, self.num_fine), device = self.device);
            # w_m, cdf, cdf_ys: (P, S - 2), (P, S - 1), (P, F)
            
            c_idx = th.searchsorted(cdf, cdf_ys, right = True);
            l_, u_ = th.max(th.zeros_like(c_idx), c_idx - 1), th.min(th.tensor(cdf.shape[-1] - 1), c_idx);
            # l_, u_: (P, F), (P, F)
            
            c_l, t_l = th.gather(cdf, 1, l_), th.gather(t_m, 1, l_);
            q_ = th.gather(cdf, 1, u_) - c_l; q_ = th.where(q_ < 1e-5, th.ones_like(q_), q_);
            t_ = th.sort(th.cat([t_, t_l + (cdf_ys - c_l) * (th.gather(t_m, 1, u_) - t_l) / q_], dim = -1), dim = -1)[0];
            dt = th.cat([t_[:, 1:] - t_[:, :-1], th.full((t_.shape[0], 1), 1e10, device = self.device)], dim = -1);
            # c_l, t_l, q_, t_, dt: (P, F), (P, F), (P, F), (P, F + S), (P, F + S)
            
            sig, rad = self.fine_net(o_.unsqueeze(1) + d_.repeat(1, t_.shape[1], 1) * t_.unsqueeze(2), d_);
            v_ = -sig.view((dt.shape[0], -1)) * dt; cv_ = th.cumsum(v_, dim = -1); cv_ = th.roll(cv_, 1, -1); cv_[:, 0] = 0.0;
            wgt = th.exp(cv_) * (1.0 - th.exp(v_));
            rgb = th.sum(wgt.unsqueeze(2) * rad, dim = 1);
            # wgt, rgb: (P, F + S), (P, 3)
        
            rgbs.append(rgb);
        
        rgb = th.cat(rgbs, dim = 0).reshape((1, h_, w_, -1)).permute(0, 3, 1, 2).clamp(0.0, 1.0);
        rig = th.from_numpy(img).to(self.device).reshape((1, h_, w_, -1)).permute(0, 3, 1, 2);
        # rgb, rig: (1, 3, H, W)
        
        lpips = self.lpips(rgb, rig).item();
        psnr = self.psnr(rgb, rig).item();
        ssim = self.ssim(rgb, rig).item();
        smp_img = (th.cat((rgb, rig), dim = 3).reshape((3, h_, -1)).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8);
        
        return lpips, psnr, ssim, smp_img;
    
    def save(self, path):
        th.save({"corsn": self.cors_net.state_dict(),
                 "finen": self.fine_net.state_dict(),
                 "optim": self.optim.state_dict(),
                 "scdlr": self.scdlr.state_dict()},
                path);
    
    def load(self, path):
        ckpt = th.load(path, map_location = "cpu");
        self.cors_net.load_state_dict(ckpt["corsn"]);
        self.fine_net.load_state_dict(ckpt["finen"]);
        self.optim.load_state_dict(ckpt["optim"]);
        self.scdlr.load_state_dict(ckpt["scdlr"]);