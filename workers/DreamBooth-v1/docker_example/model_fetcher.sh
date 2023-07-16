mkdir /src/experience-realistic
cd /src/experience-realistic
git init
git lfs install --system --skip-repo
git remote add -f origin  "https://huggingface.co/jleifeld/experience-realistic"
git config core.sparsecheckout true
echo -e "\nscheduler\ntext_encoder\ntokenizer\nunet\nvae\nmodel_index.json\n!vae/diffusion_pytorch_model.bin\n!*.safetensors" > .git/info/sparse-checkout
git pull origin main
wget -q -O vae/diffusion_pytorch_model.bin https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin
rm -r .git
rm model_index.json
wget https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/Dreambooth/model_index.json
