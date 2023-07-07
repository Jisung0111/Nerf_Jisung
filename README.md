# Basic Setup

<pre>
bash preprocess.sh
</pre>


# Training

If you don't modify any hyperparameter,
<pre>
python train.py
</pre>

Or, store hyperparameters in Exps folder as json file. (Ex. hparam0.json {"gpu": "cuda:7"})

<pre>
python train.py --hparam Exps/hparam0.json
</pre>

All results will be stored in Results/Result{result_num}

Hyperparameters: "Results/Result{result_num}/hparam.json"

Training status: "Results/Result{result_num}/train_log.txt"

Images from each evaluation: "Results/Result{result_num}/Samples/"

You can plot graphs by entering following command.

<pre>
python plot.py --result {result_num}
</pre>

# Test

You can check 200 samples from test set and evaluate lpips, psnr, ssim.

(If you are still running the training code, I recommend you to generate new hyperparameter json file to avoid GPU memory shortage. Ex. Results/Result{result_num}/hparam_test.json {..., "gpu": "cuda:8", ...})

<pre>
python reviewer.py --hparam Results/Result{result_num}/hparam.json --test 1
</pre>

Testing status: "Results/Result{result_num}/test_log.txt"

Rendered images: "Results/Result{result_num}/Tested/"

# Rendering

You can generate a video.

(If you are still running the training code, I recommend you to generate new hyperparameter json file to avoid GPU memory shortage. Ex. Results/Result{result_num}/hparam_render.json {..., "gpu": "cuda:9", ...})

<pre>
python reviewer.py --hparam Results/Result{result_num}/hparam.json --test 0
</pre>

Rendering status: "Results/Result{result_num}/render_log.txt"

Rendered images: "Results/Result{result_num}/Rendered/"

Video: "Results/Result{result_num}/rendered.mp4"
