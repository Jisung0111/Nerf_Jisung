import matplotlib.pyplot as plt
import pickle
import argparse
import numpy as np
import seaborn as sns

def plot(history, result_path):
    sns.set_theme();
    plt.figure(figsize = (18, 12));
    
    plt.subplot(2, 3, 1);
    plt.title("Coarse Loss Min: {:.6f} (Epoch: {})".format(np.min(history["cors_loss"]), np.argmin(history["cors_loss"])));
    plt.xlabel("Epoch");
    plt.ylabel("MSE Loss");
    plt.yscale("log");
    plt.plot(list(range(len(history["cors_loss"]))), history["cors_loss"]);
    
    plt.subplot(2, 3, 2);
    plt.title("Fine Loss Min: {:.6f} (Epoch: {})".format(np.min(history["fine_loss"]), np.argmin(history["fine_loss"])));
    plt.xlabel("Epoch");
    plt.ylabel("MSE Loss");
    plt.yscale("log");
    plt.plot(list(range(len(history["fine_loss"]))), history["fine_loss"]);
    
    plt.subplot(2, 3, 3);
    plt.title("Loss Min: {:.6f} (Epoch: {})".format(np.min(history["loss"]), np.argmin(history["loss"])));
    plt.xlabel("Epoch");
    plt.ylabel("MSE Loss");
    plt.ylim([0.00, 0.01]);
    plt.plot(list(range(len(history["loss"]))), history["loss"]);
    
    plt.subplot(2, 3, 4);
    plt.title("LPIPS Min: {:.4f} (Epoch: {})".format(np.min(history["lpips"]), history["epoch"][np.argmin(history["lpips"])]));
    plt.xlabel("Epoch");
    plt.ylabel("LPIPS");
    plt.ylim([0.04, 0.12]);
    plt.plot([0, history["epoch"][-1]], [0.06, 0.06], color = "black");
    plt.plot(history["epoch"], history["lpips"]);
    
    plt.subplot(2, 3, 5);
    plt.title("PSNR Max: {:.2f} (Epoch: {})".format(np.max(history["psnr"]), history["epoch"][np.argmax(history["psnr"])]));
    plt.xlabel("Epoch");
    plt.ylabel("PSNR");
    plt.ylim([25, 30]);
    plt.plot([0, history["epoch"][-1]], [26, 26], color = "black");
    plt.plot(history["epoch"], history["psnr"]);
    
    plt.subplot(2, 3, 6);
    plt.title("SSIM Max: {:.4f} (Epoch: {})".format(np.max(history["ssim"]), history["epoch"][np.argmax(history["ssim"])]));
    plt.xlabel("Epoch");
    plt.ylabel("SSIM");
    plt.ylim([0.88, 0.94]);
    plt.plot([0, history["epoch"][-1]], [0.9, 0.9], color = "black");
    plt.plot(history["epoch"], history["ssim"]);
    
    plt.savefig(result_path + "statistic.png", dpi = 300);

if __name__ == "__main__":
    parser = argparse.ArgumentParser();
    parser.add_argument('--result');
    file_loc = parser.parse_args();
    
    result_path = 'Results/Result' + file_loc.result + '/';
    with open(result_path + 'history.pkl', 'rb') as f:
        history = pickle.load(f);
    
    plot(history, result_path);